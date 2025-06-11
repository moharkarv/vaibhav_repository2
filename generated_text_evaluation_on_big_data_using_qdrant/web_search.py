import re, gc
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests

from langchain.agents import initialize_agent, AgentExecutor, tool
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage
from serpapi import GoogleSearch
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
noise = str(uuid.uuid4())[:8]



class FactCheckerAgent:
    def __init__(self, llm: ChatGoogleGenerativeAI, gemini_api_key: str, serp_api_key: str):
        genai.configure(api_key=gemini_api_key)  # ✅ Explicit Gemini key

        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.serp_api_key = serp_api_key
        self.agent_executor = self._initialize_agent()

    @tool
    def search_web(query: str) -> dict:
        """Searches the web using SerpAPI and checks content relevance with LLM."""
        params = {
            "engine": "google",
            "q": query,
            "api_key": FactCheckerAgent.serp_api_key_static,
            "num": 5
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" not in results:
            return {"error": " No search results found."}

        for idx, result in enumerate(results["organic_results"][:5]):
            top_link = result.get("link")

            try:
                response = requests.get(top_link, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                main_content = soup.find('article') or soup.find('main') or soup
                paragraphs = main_content.find_all("p")
                extracted_text = "\n".join([p.get_text().strip()
                                            for p in paragraphs
                                            if len(p.get_text().strip()) > 50])

                if not extracted_text:
                    continue

                prompt = f"""Analyze if this content answers "{query}". Consider:
                - Direct mentions
                - Contextual relevance
                - Temporal relevance

                Content: {extracted_text[:1500]}

                Respond ONLY with 'YES' or 'NO':"""

                llm_response = FactCheckerAgent.llm_static.invoke(prompt)
                response_text = (
                    llm_response.content.lower()
                    if isinstance(llm_response, AIMessage)
                    else str(llm_response).lower()
                )

                if re.search(r'\b(yes|yeah|yep|correct)\b', response_text):
                    return {
                        "status": "Relevant content found",
                        "source": top_link,
                        "content": extracted_text[:5000]
                    }

            except requests.RequestException:
                continue

        return {"error": " No relevant content found after 5 attempts."}

    @tool
    def search_wikipedia_api(query: str) -> dict:
        """Fetch Wikipedia summary using the MediaWiki API."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",  #"extracts" means: “Give me the plain text content of the article.”
            "exintro": True,     #Return only the introduction of the article — everything before the first heading.
            "explaintext": True, #Returns the extract as plain text, without any HTML formatting.
            "titles": query, 
            "redirects": 1      #Automatically follows redirects.Forexample, if someone searches "Einstein", Wikipedia may redirect it to "Albert Einstein" — this makes sure you get the final page.
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for _, page in pages.items():
                if "extract" in page:
                    return {
                        "status": "Wikipedia content found",
                        "title": page.get("title"),
                        "summary": page.get("extract")
                    }

            return {"error": "No content found on Wikipedia."}

        except Exception as e:
            return {"error": str(e)}

    @tool
    def do_nothing(query: str) -> dict:
        """A fallback tool for non-actionable queries."""
        return {"status": "No action needed", "message": "The query does not require searching."}

    def _initialize_agent(self):
        # Static references for tools
        FactCheckerAgent.llm_static = self.llm
        FactCheckerAgent.serp_api_key_static = self.serp_api_key

        tools = [
            Tool.from_function(func=self.search_web, name="search_web", description="Use this to search the web."),
            Tool.from_function(func=self.search_wikipedia_api, name="search_wikipedia_api", description="Use this to search Wikipedia."),
            Tool.from_function(func=self.do_nothing, name="do_nothing", description="Use when no action is needed.")
        ]

        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent.agent,
            tools=agent.tools,
            #memory=self.memory,
            memory=None,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

    def run_fact_check(self, question: str, answer: str):
        query = (
            "Fact-check the following answer.\n\n"
            f"text: {answer}\n\n"
            "Instructions:\n"
            "- Check factual correctness of the answer.\n"
            "- Use reliable sources (Wikipedia or the web) if needed.\n"
            "- Point out any inaccuracies.\n"
            "- Provide the correct information if something is wrong.\n"
            "- Return a factual accuracy score out of 100."
        )

        result = self.agent_executor.invoke({"input": query})
        return result.get("output", "").strip()

    def cleanup(self):
        """Free up memory after usage."""
        del self.agent_executor
        del self.memory
        del self.llm
        gc.collect()






#Zero-shot	No examples needed
#ReAct	Uses "Thought → Action → Observation" reasoning
#Description	Agent decides tool usage based on your tool descriptions



# 🐛 verbose=True
# Prints everything happening inside the agent:

# Thoughts

# Actions

# Observations

# Final answer

# Helps a lot for debugging and seeing how the LLM thinks.

# 🛡️ handle_parsing_errors=True
# If the LLM returns a malformed output (e.g., bad JSON), LangChain won't crash — it will gracefully handle the error.

# Super useful when LLM outputs are unpredictable or occasionally incorrect in format.