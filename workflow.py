from typing import Dict, Any
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step, Event
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from groq import Groq
from tools import query_engine, analyze_response, generate_final_answer, web_search_tool, handle_human_message, classify_query
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputEvent(Event):
    input: str

class ToolCallEvent(Event):
    id: str
    name: str
    params: Dict[str, Any]

class UserDecisionEvent(Event):
    decision: str

class RAGSystem(Workflow):
    def __init__(
        self,
        *args: Any,
        index: VectorStoreIndex,
        llm: Any,
        timeout: int = 20,
    ):
        super().__init__(*args)
        self._timeout = timeout
        self.llm = llm
        self.index = index

        self.tools = {
            "query_engine": FunctionTool(
                fn=lambda **params: query_engine(self.index, **params),
                metadata={"name": "query_engine", "description": "Fetches results from the vector index based on the query."}
            ),
            "analyze_response": FunctionTool(
                fn=lambda **params: analyze_response(self.llm, **params),
                metadata={"name": "analyze_response", "description": "Analyzes the response and determines if more information is needed."}
            ),
            "generate_final_answer": FunctionTool(
                fn=lambda **params: generate_final_answer(self.llm, **params),
                metadata={"name": "generate_final_answer", "description": "Generates the final answer based on the analysis."}
            ),
            "web_search_tool": FunctionTool(
                fn=lambda **params: web_search_tool(**params),
                metadata={"name": "web_search_tool", "description": "Searches the web for the query."}
            )
        }

    @step
    async def handle_initial_query(self, ev: InputEvent, context: Dict[str, Any]) -> StopEvent | ToolCallEvent:
        query = ev.input
        context["original_query"] = query

        try:
            classification = await classify_query(self.llm, query)
            logger.info(f"Query classification: {classification}")

            if classification == "greeting":
                logger.info("Detected a human message. Invoking LLM agent")
                response = await handle_human_message(self.llm, query)
                logger.info(f"LLM response: {response['response']}")
                return StopEvent(result={"message": response['response']})

            elif classification == "index_search":
                return ToolCallEvent(
                    id="initial_query",
                    name="query_engine",
                    params={"query": query},
                )
            else:
                logger.info("Query is not related to the context. Initiating web search")
                return ToolCallEvent(
                    id="web_search",
                    name="web_search_tool",
                    params={"query": query}
                )
        except Exception as e:
            logger.error(f"Error in handle_initial_query: {e}")
            return StopEvent(result={"error": str(e)})

    @step
    async def process_initial_response(self, ev: ToolCallEvent, context: Dict[str, Any]) -> StopEvent | ToolCallEvent:
        if isinstance(ev, StopEvent):
            logger.info("Received StopEvent, no further processing needed.")
            return ev
        
        try:
            tool_output = await self.tools[ev.name].fn(**ev.params)
            initial_response = tool_output["response"]
            context["initial_response"] = initial_response
            logger.info(f"Initial Response: {initial_response}")

            if ev.id == "web_search":
                return StopEvent(result={"message": "Web search results provided.", "response": initial_response})

            # Return the initial response to be presented to the user
            return StopEvent(result={"message": "Initial response", "response": initial_response, "requires_feedback": True})
        except Exception as e:
            logger.error(f"Error in process_initial_response: {e}")
            return StopEvent(result={"error": str(e)})

    @step
    async def handle_user_decision(self, ev: UserDecisionEvent, context: Dict[str, Any]) -> StopEvent | ToolCallEvent:
        try:
            if ev.decision == 'yes':
                return StopEvent(result={"message": "User confirmed satisfactory response.", "response": context["initial_response"]})
            else:
                logger.info("Starting Advanced Query")
                return ToolCallEvent(
                    id="analyze_response",
                    name="analyze_response",
                    params={
                        "initial_response": context["initial_response"], 
                        "query": context["original_query"]
                    },
                )
        except Exception as e:
            logger.error(f"Error in handling user decision: {e}")
            return StopEvent(result={"error": str(e)})

    @step
    async def process_analysis(self, ev: ToolCallEvent, context: Dict[str, Any]) -> ToolCallEvent:
        try:
            analysis_output = await self.tools[ev.name].fn(**ev.params)
            analysis = analysis_output["analysis"]
            logger.info(f"RAGSystem: Analysis result: {analysis}")
            context["follow_up_query"] = analysis
            
            return ToolCallEvent(
                id="additional_query",
                name="query_engine",
                params={"query": analysis},
            )
        except Exception as e:
            logger.error(f"Error in process_analysis: {e}")
            return StopEvent(result={"error": str(e)})

    @step
    async def fetch_additional_info(self, ev: ToolCallEvent, context: Dict[str, Any]) -> ToolCallEvent:
        try:
            additional_info = await self.tools[ev.name].fn(**ev.params)
            logger.info(f"RAGSystem: Additional info retrieved: {additional_info['response']}")
            context["additional_info"] = additional_info["response"]
            
            return ToolCallEvent(
                id="generate_final_answer",
                name="generate_final_answer",
                params={
                    "query": context["original_query"],
                    "initial_response": context["initial_response"],
                    "additional_info": additional_info["response"]
                }
            )
        except Exception as e:
            logger.error(f"Error in fetch_additional_info: {e}")
            return StopEvent(result={"error": str(e)})

    @step
    async def generate_final_answer(self, ev: ToolCallEvent, context: Dict[str, Any]) -> StopEvent:
        try:
            final_answer = await self.tools[ev.name].fn(**ev.params)
            return StopEvent(result={"response": final_answer["final_answer"]})
        except Exception as e:
            logger.error(f"Error in generate_final_answer: {e}")
            return StopEvent(result={"error": str(e)})

    async def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            initial_query_event = await self.handle_initial_query(InputEvent(input=query), context)
            if isinstance(initial_query_event, StopEvent):
                return initial_query_event.result

            process_initial_response_event = await self.process_initial_response(initial_query_event, context)
            if isinstance(process_initial_response_event, StopEvent):
                return process_initial_response_event.result

            # If we reach this point, it means we need user feedback
            return {"message": "Awaiting user feedback", "response": context["initial_response"], "requires_feedback": True}

        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            return {"error": str(e)}

    async def process_user_feedback(self, user_decision: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            user_decision_event = UserDecisionEvent(decision=user_decision)
            handle_user_decision_event = await self.handle_user_decision(user_decision_event, context)

            if isinstance(handle_user_decision_event, StopEvent):
                return handle_user_decision_event.result

            analysis_event = await self.process_analysis(handle_user_decision_event, context)
            if isinstance(analysis_event, StopEvent):
                return analysis_event.result

            additional_info_event = await self.fetch_additional_info(analysis_event, context)
            if isinstance(additional_info_event, StopEvent):
                return additional_info_event.result

            final_answer_event = await self.generate_final_answer(additional_info_event, context)
            return final_answer_event.result

        except Exception as e:
            logger.error(f"Error in process_user_feedback: {e}")
            return {"error": str(e)}