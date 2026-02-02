from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings

warnings.filterwarnings("ignore")

class PostGameAgent:
    """
    A tactical analyst for post-round or general strategic queries. 
    It evaluates claims and explains trade-offs between different tactical approaches.
    """
    def __init__(self, model="llama3.2:1b", temperature=0):
        """
        Initializes the PostGameAgent with a specific LLM and conversational memory.

        :param model: The name of the Ollama model to use.
        :param temperature: The temperature for LLM generation.
        """
        # LLM
        self.llm = Ollama(
            model=model,
            temperature=temperature
        )

        # Memory (conversation continuity only)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="claim",
            return_messages=False
        )

        # Prompt
        self.prompt = PromptTemplate(
            input_variables=["claim", "chat_history", "data_history"],
            template="""
You are an AI Assistant Coach specialized exclusively in professional VALORANT.

You operate as a data-driven analyst embedded within a professional coaching staff.
Your purpose is to help teams choose between tactical options based on statistical insights,
not to dictate a single correct answer.

DATA HISTORY (GRID Snapshots)
{data_history}

RULE ABOUT MEMORY
- Chat history exists only to maintain conversational continuity.
- You MUST NOT treat past claims as evidence for the current claim.
- Each claim must be evaluated independently.

CHAT HISTORY (context only)
{chat_history}

INPUT CLAIM
{claim}

YOUR CORE TASK
- Interpret the claim.
- Evaluate how actionable it is.
- Help the team choose between viable tactical options by explaining tradeoffs.

OUTPUT RULES (STRICT)
- NEVER state a single “best” decision.
- ALWAYS present at least TWO viable decision options.
- EVERY option must include:
  - Expected upside
  - Explicit risk or tradeoff
  - Context where it makes sense
- If sample size, map, side, or timing is missing:
  - Downgrade confidence.
  - Do NOT fabricate missing data.
- If the claim is weak, low-sample, or poorly specified:
  - State that no strong conclusion can be drawn.

OUTPUT FORMAT (REQUIRED)

INTERPRETATION
- One short paragraph explaining what the claim suggests and its limitations.

DECISION OPTIONS
Option A:
- Expected upside:
- Risk / tradeoff:
- Best used when:

Option B:
- Expected upside:
- Risk / tradeoff:
- Best used when:

(Option C only if clearly justified)

CONFIDENCE
- High / Medium / Low
- One sentence explaining why.

FAILURE CONDITION
If the input sentence lacks enough information to reasonably evaluate decision options:
- Say explicitly that no actionable decision can be made.
- Do not guess.
- Do not fill gaps.

OBJECTIVE
Improve expected round win probability by clarifying tradeoffs behind tactical choices,
while keeping final authority with the coaching staff.
"""
        )

        # Chain (LCEL)
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables(x)["chat_history"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # Ask function
    def ask(self, claim: str, data_history: str = "No data history available."):
        """
        Analyzes a strategic claim and provides a detailed analysis of tradeoffs.

        :param claim: The strategic claim or question to evaluate.
        :param data_history: The history of snapshots from the GRID pipeline.
        :return: A string containing the analysis, decision options, and confidence level.
        """
        inputs = {"claim": claim, "data_history": data_history}
        response = self.chain.invoke(inputs)
        
        # Manually save to memory
        self.memory.save_context({"claim": claim}, {"output": response})
        
        return response
