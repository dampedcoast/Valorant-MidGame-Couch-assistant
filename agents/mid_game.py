from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings

warnings.filterwarnings("ignore")


class MidGameAgent:
    """
    A live-round tactical advisor that provides exactly two actionable options 
    for the IGL (In-Game Leader) under strict time constraints.
    """
    def __init__(self, model="llama3.2:1b", temperature=0):
        """
        Initializes the MidGameAgent with a specific LLM and short-term memory.

        :param model: The name of the Ollama model to use.
        :param temperature: The temperature for LLM generation.
        """
        # LLM
        self.llm = Ollama(model=model, temperature=temperature)

        # Memory (short-term, session only)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=False
        )

        # Prompt
        self.prompt = PromptTemplate(
            input_variables=["round_data", "question", "chat_history"],
            template= """
            You are a Mid-Round Decision Support AI for professional VALORANT.
            
            You assist the IGL during a LIVE ROUND under time pressure.
            You do NOT coach, explain theory, analyze history, or review past rounds.
            
            SCOPE RULES (STRICT)
            - This is MID-ROUND only.
            - Focus on the NEXT 5–15 seconds of play.
            - Use imperative, concise language.
            - No long explanations.
            - No hindsight.
            - No motivational or educational talk.
            
            MEMORY RULE
            - Chat history exists ONLY for conversational continuity.
            - You MUST NOT use chat history as tactical evidence.
            
            CHAT HISTORY (context only, NOT evidence)
            {chat_history}
            
            ROUND DATA (SINGLE SOURCE OF TRUTH)
            {round_data}
            
            USER QUESTION
            {question}
            
            HARD OUTPUT CONSTRAINTS
            - You MUST output EXACTLY TWO options.
            - No more, no less.
            - Every option must be immediately actionable.
            
            OUTPUT FORMAT (REQUIRED)
            
            SITUATION SNAPSHOT
            - One sentence summarizing the current round state.
            
            OPTIONS
            Option A:
            - Immediate action:
            - Upside:
            - Primary risk:
            
            Option B:
            - Immediate action:
            - Upside:
            - Primary risk:
            
            RISK FLAG
            - One short warning describing the most likely way the round fails in the next moments.
            
            FAILURE CONDITION (OVERRIDES ALL)
            If the round data is unclear, missing, or insufficient for a live decision:
            - Output ONLY this sentence:
            “Insufficient live information.”
            - Do NOT add anything else.
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
    def ask(self, round_data, question):
        """
        Generates tactical options based on the provided round data and user question.

        :param round_data: Current game state information (single source of truth).
        :param question: The user's query about what to do.
        :return: A string containing exactly two tactical options or a failure message.
        """
        inputs = {"round_data": round_data, "question": question}
        response = self.chain.invoke(inputs)
        
        # Manually save to memory as we are using LCEL without a memory-integrated runnable
        self.memory.save_context({"question": question}, {"output": response})
        
        return response




def main():
    #example usage:
    agent = MidGameAgent()
    data = """Time: 28s
Side: Attack
Alive (Us/Them): 3 / 4
Spike: Dropped mid
Utility (Us): 1 smoke, 0 flashes, 1 molly
Ults (Us): None
Map Control:
  Mid: Lost
  A Main: Unknown
  B Main: Controlled
Economy: Rifle round
IGL Constraint: Save if plant impossible"""
    print(agent.ask(data, 'what we do'))


if __name__ == "__main__":
    main()
