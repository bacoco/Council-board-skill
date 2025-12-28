"""
PersonaManager - Dynamic persona assignment based on question type.

Replaces hardcoded persona mappings with intelligent assignment.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
import re


class QuestionType(Enum):
    """Types of questions for persona assignment."""
    FACTUAL = "factual"              # Facts, history, geography
    TECHNICAL = "technical"           # Code, architecture, engineering  
    ETHICAL = "ethical"               # Morality, philosophy, dilemmas
    FINANCIAL = "financial"           # Economics, crypto, business
    CREATIVE = "creative"             # Art, design, innovation
    SCIENTIFIC = "scientific"         # Physics, biology, research
    STRATEGIC = "strategic"           # Planning, decision-making
    ANALYTICAL = "analytical"         # Data, statistics, analysis


@dataclass
class Persona:
    """Persona definition for council member."""
    title: str
    role: str
    prompt_prefix: str
    specializations: List[str]


# Persona library organized by category
PERSONA_LIBRARY = {
    "historian": Persona(
        title="Historian",
        role="Historical context and cultural analysis",
        prompt_prefix="You are a Historian. Provide historical context, cultural perspectives, and analyze how similar questions have been answered throughout history.",
        specializations=["history", "culture", "precedent", "evolution"]
    ),
    "scientist": Persona(
        title="Scientist",
        role="Scientific analysis and empirical reasoning",
        prompt_prefix="You are a Scientist. Focus on empirical evidence, scientific method, and data-driven analysis. Question assumptions and seek verifiable facts.",
        specializations=["empiricism", "methodology", "evidence", "research"]
    ),
    "ethicist": Persona(
        title="Ethicist",
        role="Moral reasoning and ethical frameworks",
        prompt_prefix="You are an Ethicist. Analyze the moral dimensions, apply ethical frameworks (utilitarianism, deontology, virtue ethics), and consider stakeholder impacts.",
        specializations=["morality", "ethics", "justice", "values"]
    ),
    "economist": Persona(
        title="Economist",
        role="Economic analysis and incentive structures",
        prompt_prefix="You are an Economist. Analyze incentive structures, market dynamics, cost-benefit trade-offs, and long-term economic implications.",
        specializations=["markets", "incentives", "trade-offs", "efficiency"]
    ),
    "technologist": Persona(
        title="Technologist",
        role="Technical feasibility and innovation",
        prompt_prefix="You are a Technologist. Assess technical feasibility, identify innovation opportunities, and evaluate technological trade-offs.",
        specializations=["technology", "innovation", "implementation", "scalability"]
    ),
    "architect": Persona(
        title="Systems Architect",
        role="System design and architecture",
        prompt_prefix="You are a Systems Architect. Focus on structural design, modularity, maintainability, and long-term architectural implications.",
        specializations=["architecture", "design", "structure", "maintainability"]
    ),
    "pragmatist": Persona(
        title="Pragmatist",
        role="Practical implementation and real-world constraints",
        prompt_prefix="You are a Pragmatist. Focus on what actually works in practice, consider implementation constraints, and prioritize operational feasibility.",
        specializations=["practicality", "implementation", "feasibility", "operations"]
    ),
    "contrarian": Persona(
        title="Contrarian Thinker",
        role="Challenge assumptions and conventional wisdom",
        prompt_prefix="You are a Contrarian Thinker. Challenge popular assumptions, question conventional wisdom, and propose alternative perspectives that others might overlook.",
        specializations=["critical thinking", "assumptions", "alternatives", "skepticism"]
    ),
    "systems_thinker": Persona(
        title="Systems Thinker",
        role="Holistic analysis and emergent properties",
        prompt_prefix="You are a Systems Thinker. Analyze second-order effects, feedback loops, emergent properties, and how components interact in complex systems.",
        specializations=["systems", "complexity", "emergence", "feedback loops"]
    ),
    "legal_scholar": Persona(
        title="Legal Scholar",
        role="Legal frameworks and regulatory analysis",
        prompt_prefix="You are a Legal Scholar. Analyze legal implications, regulatory frameworks, precedent, and compliance considerations.",
        specializations=["law", "regulation", "compliance", "precedent"]
    ),
    "data_analyst": Persona(
        title="Data Analyst",
        role="Statistical analysis and pattern recognition",
        prompt_prefix="You are a Data Analyst. Focus on statistical patterns, data-driven insights, quantitative analysis, and evidence-based reasoning.",
        specializations=["statistics", "data", "patterns", "metrics"]
    ),
    "creative_strategist": Persona(
        title="Creative Strategist",
        role="Innovation and unconventional solutions",
        prompt_prefix="You are a Creative Strategist. Think outside the box, propose innovative solutions, and explore unconventional approaches.",
        specializations=["creativity", "innovation", "brainstorming", "unconventional"]
    )
}


class PersonaManager:
    """Manages dynamic persona assignment based on question type."""
    
    def __init__(self):
        self.persona_library = PERSONA_LIBRARY
    
    def analyze_query_type(self, query: str) -> QuestionType:
        """
        Analyze query to determine question type.
        
        Uses keyword matching and pattern recognition.
        """
        query_lower = query.lower()
        
        # Factual indicators
        factual_patterns = [
            r'\b(what is|who is|where is|when did|capital of|history of)\b',
            r'\b(fact|date|location|name of)\b'
        ]
        if any(re.search(p, query_lower) for p in factual_patterns):
            return QuestionType.FACTUAL
        
        # Ethical indicators
        ethical_patterns = [
            r'\b(should|ought|moral|ethical|right|wrong|trolley problem)\b',
            r'\b(justice|fair|virtue|duty)\b'
        ]
        if any(re.search(p, query_lower) for p in ethical_patterns):
            return QuestionType.ETHICAL
        
        # Financial/Economic indicators
        financial_patterns = [
            r'\b(crypto|bitcoin|invest|market|economic|financial|money|price)\b',
            r'\b(trade|stock|currency|blockchain)\b'
        ]
        if any(re.search(p, query_lower) for p in financial_patterns):
            return QuestionType.FINANCIAL
        
        # Technical indicators
        technical_patterns = [
            r'\b(code|implement|architecture|database|algorithm|optimize|debug)\b',
            r'\b(api|framework|library|performance|scalability)\b'
        ]
        if any(re.search(p, query_lower) for p in technical_patterns):
            return QuestionType.TECHNICAL
        
        # Scientific indicators
        scientific_patterns = [
            r'\b(scientific|research|study|evidence|hypothesis|experiment)\b',
            r'\b(prove|disprove|theory|empirical)\b'
        ]
        if any(re.search(p, query_lower) for p in scientific_patterns):
            return QuestionType.SCIENTIFIC
        
        # Strategic indicators
        strategic_patterns = [
            r'\b(strategy|prioritize|decision|choose|plan|approach)\b',
            r'\b(roadmap|versus|vs|better)\b'
        ]
        if any(re.search(p, query_lower) for p in strategic_patterns):
            return QuestionType.STRATEGIC
        
        # Default to analytical
        return QuestionType.ANALYTICAL
    
    def assign_personas(self, query: str, num_models: int = 3) -> List[Persona]:
        """
        Assign personas dynamically based on query type.
        
        Returns list of personas suited to the question.
        """
        query_type = self.analyze_query_type(query)
        
        # Persona assignment rules by question type
        assignments = {
            QuestionType.FACTUAL: ["historian", "scientist", "data_analyst"],
            QuestionType.ETHICAL: ["ethicist", "legal_scholar", "systems_thinker"],
            QuestionType.FINANCIAL: ["economist", "technologist", "data_analyst"],
            QuestionType.TECHNICAL: ["architect", "technologist", "pragmatist"],
            QuestionType.SCIENTIFIC: ["scientist", "data_analyst", "contrarian"],
            QuestionType.STRATEGIC: ["systems_thinker", "pragmatist", "contrarian"],
            QuestionType.CREATIVE: ["creative_strategist", "contrarian", "systems_thinker"],
            QuestionType.ANALYTICAL: ["data_analyst", "scientist", "systems_thinker"]
        }
        
        persona_keys = assignments.get(query_type, ["systems_thinker", "pragmatist", "contrarian"])
        
        # Return personas, cycling if needed
        personas = []
        for i in range(num_models):
            key = persona_keys[i % len(persona_keys)]
            personas.append(self.persona_library[key])
        
        return personas
    
    def get_persona_for_model(self, query: str, model_index: int) -> Persona:
        """Get specific persona for a model by index."""
        personas = self.assign_personas(query)
        return personas[model_index % len(personas)]


# Example usage
if __name__ == "__main__":
    pm = PersonaManager()
    
    # Test different question types
    queries = [
        "What is the capital of France?",
        "Should we sacrifice one to save five?",
        "Is cryptocurrency a revolutionary technology?",
        "How do I optimize this database query?",
        "What are the effects of climate change?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Type: {pm.analyze_query_type(query).value}")
        personas = pm.assign_personas(query)
        for i, persona in enumerate(personas):
            print(f"  Model {i+1}: {persona.title} - {persona.role}")
