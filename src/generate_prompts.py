"""
Generate a diverse bank of prompts for Opus style analysis.

Weighted toward technical/professional writing with variety in length,
formality, and specificity.
"""

import json
import random
from pathlib import Path
from typing import Iterator

# Random seed for reproducibility
random.seed(42)

# Prompt templates by category
# Weighted toward technical/professional content

TECHNICAL_EXPLANATIONS = [
    # Programming
    "Explain how {concept} works in {language}.",
    "What's the difference between {a} and {b} in programming?",
    "Describe the tradeoffs of using {pattern} in software architecture.",
    "How would you implement {feature} in a production system?",
    "Explain {algorithm} and when you'd use it.",
    "What are best practices for {topic} in {language}?",
    "Walk through how {technology} handles {operation}.",
    "Compare {framework_a} vs {framework_b} for {use_case}.",
    # Science/Math
    "Explain {scientific_concept} in terms a smart colleague would understand.",
    "What is {mathematical_concept} and why does it matter?",
    "Describe how {process} works at a technical level.",
    "What's the relationship between {concept_a} and {concept_b}?",
    # Systems
    "How does {system} achieve {property}?",
    "Explain the architecture of a typical {system_type}.",
    "What happens when {event} occurs in {context}?",
    "Describe the lifecycle of {entity} in {system}.",
]

TECHNICAL_FILLS = {
    "concept": ["garbage collection", "async/await", "dependency injection",
                "closures", "recursion", "hashing", "indexing", "caching",
                "connection pooling", "rate limiting", "load balancing"],
    "language": ["Python", "JavaScript", "Go", "Rust", "TypeScript", "Java"],
    "a": ["threads", "let", "composition", "REST", "SQL", "mocking"],
    "b": ["processes", "const", "inheritance", "GraphQL", "NoSQL", "stubbing"],
    "pattern": ["microservices", "event sourcing", "CQRS", "repository pattern",
                "factory pattern", "singleton", "observer pattern"],
    "feature": ["authentication", "real-time notifications", "search",
                "file uploads", "caching layer", "rate limiting", "logging"],
    "algorithm": ["quicksort", "binary search", "BFS", "dynamic programming",
                  "consistent hashing", "bloom filters", "Dijkstra's algorithm"],
    "topic": ["error handling", "testing", "logging", "configuration management",
              "database migrations", "API versioning", "security"],
    "technology": ["PostgreSQL", "Redis", "Kafka", "Docker", "Kubernetes", "nginx"],
    "operation": ["transactions", "replication", "failover", "scaling", "routing"],
    "framework_a": ["React", "Django", "FastAPI", "Express", "Spring"],
    "framework_b": ["Vue", "Flask", "Rails", "NestJS", "Micronaut"],
    "use_case": ["large applications", "rapid prototyping", "high-performance APIs",
                 "real-time applications", "data-intensive applications"],
    "scientific_concept": ["entropy", "natural selection", "the uncertainty principle",
                          "neural plasticity", "protein folding", "CRISPR"],
    "mathematical_concept": ["eigenvalues", "Bayesian inference", "gradient descent",
                            "the central limit theorem", "Fourier transforms"],
    "process": ["TCP handshake", "DNS resolution", "TLS negotiation",
                "memory allocation", "query optimization", "compilation"],
    "concept_a": ["latency", "consistency", "precision", "bias"],
    "concept_b": ["throughput", "availability", "recall", "variance"],
    "system": ["distributed database", "message queue", "CDN", "search engine"],
    "property": ["consistency", "fault tolerance", "low latency", "high throughput"],
    "system_type": ["web application", "data pipeline", "ML training system",
                    "real-time analytics platform", "content delivery network"],
    "event": ["a node fails", "network partition occurs", "memory runs out",
              "a deadlock happens", "cache invalidation triggers"],
    "context": ["a distributed system", "a database", "a container orchestrator",
                "a message broker", "a load balancer"],
    "entity": ["a request", "a transaction", "a connection", "a message", "a job"],
}

ANALYSIS_OPINION = [
    "What are the pros and cons of {topic}?",
    "Evaluate {approach} for {context}.",
    "When would you choose {option_a} over {option_b}?",
    "What are the most common mistakes when {activity}?",
    "Critique the approach of {methodology}.",
    "What factors should you consider when {decision}?",
    "Is {claim} true? Analyze the evidence.",
    "What are the hidden costs of {choice}?",
    "Compare the tradeoffs between {tradeoff_a} and {tradeoff_b}.",
    "What's overrated and underrated about {topic}?",
]

ANALYSIS_FILLS = {
    "topic": ["microservices", "TypeScript", "NoSQL databases", "serverless",
              "test-driven development", "pair programming", "agile methodology",
              "monorepos", "GraphQL", "Kubernetes"],
    "approach": ["using ORMs", "premature optimization", "trunk-based development",
                 "feature flags", "blue-green deployments", "strangler fig pattern"],
    "context": ["a startup", "enterprise software", "a small team", "legacy systems",
                "greenfield projects", "high-traffic applications"],
    "option_a": ["SQL", "monolith", "REST", "polling", "synchronous processing"],
    "option_b": ["NoSQL", "microservices", "GraphQL", "webhooks", "async processing"],
    "activity": ["designing APIs", "writing tests", "code reviews", "estimating tasks",
                 "debugging production issues", "onboarding new developers"],
    "methodology": ["Scrum", "waterfall", "domain-driven design", "hexagonal architecture",
                    "clean architecture", "event-driven architecture"],
    "decision": ["choosing a database", "selecting a framework", "deciding team structure",
                 "prioritizing technical debt", "choosing build vs buy"],
    "claim": ["premature optimization is the root of all evil",
              "you should always write tests first",
              "microservices solve scaling problems",
              "code comments are a code smell"],
    "choice": ["using a framework", "adding abstraction layers", "hiring specialists",
               "adopting new technology", "rewriting from scratch"],
    "tradeoff_a": ["speed of development", "flexibility", "simplicity", "consistency"],
    "tradeoff_b": ["maintainability", "performance", "correctness", "availability"],
}

INSTRUCTION_HOWTO = [
    "How do you {task}?",
    "Walk me through {process}.",
    "What's the step-by-step process for {activity}?",
    "Explain how to {action} effectively.",
    "What's your approach to {challenge}?",
    "How should I structure {artifact}?",
    "Guide me through setting up {system}.",
    "What's the best way to {goal}?",
    "How do you troubleshoot {problem}?",
    "Describe how to {operation} in {context}.",
]

INSTRUCTION_FILLS = {
    "task": ["debug a memory leak", "optimize a slow query", "refactor legacy code",
             "design a REST API", "set up CI/CD", "conduct a code review",
             "write a technical spec", "estimate a project", "run a postmortem"],
    "process": ["deploying to production", "database migration", "performance profiling",
                "security audit", "incident response", "capacity planning"],
    "activity": ["designing a system", "breaking down a large task", "writing documentation",
                 "onboarding to a new codebase", "preparing for a technical interview"],
    "action": ["write error messages", "name variables", "organize code",
               "handle edge cases", "write commit messages", "document decisions"],
    "challenge": ["managing technical debt", "handling scope creep", "balancing speed and quality",
                  "dealing with unclear requirements", "working with legacy systems"],
    "artifact": ["a technical design document", "an API", "a database schema",
                 "a test suite", "a monitoring dashboard", "error handling"],
    "system": ["a local development environment", "a CI/CD pipeline", "monitoring",
               "a staging environment", "database replication", "feature flags"],
    "goal": ["learn a new programming language", "improve code review feedback",
             "reduce deployment risk", "increase test coverage", "improve performance"],
    "problem": ["a production outage", "flaky tests", "slow builds",
                "memory issues", "race conditions", "authentication failures"],
    "operation": ["rollback a deployment", "restore from backup", "scale horizontally",
                  "implement caching", "add observability"],
    "context": ["a microservices architecture", "a monolithic application",
                "a high-traffic system", "a data pipeline", "a real-time system"],
}

CREATIVE_WRITING = [
    "Write a short scene where {scenario}.",
    "Describe {subject} in vivid detail.",
    "Write the opening paragraph of {genre} story about {topic}.",
    "Create a metaphor that explains {concept}.",
    "Write a brief character sketch of {character_type}.",
    "Describe the atmosphere of {setting}.",
    "Write dialogue between {person_a} and {person_b} about {topic}.",
    "Craft a compelling hook for an article about {subject}.",
]

CREATIVE_FILLS = {
    "scenario": ["two engineers disagree about architecture",
                 "someone discovers a critical bug before launch",
                 "a developer's side project becomes unexpectedly popular",
                 "a team celebrates shipping a difficult feature"],
    "subject": ["the feeling of finally fixing a difficult bug",
                "the chaos of a production incident",
                "the quiet focus of deep work",
                "the moment before pressing deploy"],
    "genre": ["a thriller", "a mystery", "a comedy", "a drama"],
    "topic": ["artificial intelligence", "a startup", "open source",
              "remote work", "a data breach", "technical debt"],
    "concept": ["technical debt", "distributed systems", "machine learning",
                "version control", "the cloud", "recursion"],
    "character_type": ["a burned-out senior engineer", "an enthusiastic junior developer",
                       "a pragmatic tech lead", "a visionary founder"],
    "setting": ["a late-night debugging session", "a tense product launch",
                "a quiet office after everyone's gone home", "a chaotic standup meeting"],
    "person_a": ["a developer", "a product manager", "a designer", "a CEO"],
    "person_b": ["their manager", "a skeptical colleague", "a demanding client", "an investor"],
}

CASUAL_CONVERSATION = [
    "What do you think about {topic}?",
    "How would you explain {concept} to a friend?",
    "What's your take on {debate}?",
    "Why do people {behavior}?",
    "What makes {thing} interesting?",
    "Share your thoughts on {subject}.",
    "What's underappreciated about {topic}?",
    "If you had to {hypothetical}, what would you do?",
]

CASUAL_FILLS = {
    "topic": ["the current state of web development", "AI-generated code",
              "remote work culture", "tech interviews", "open source sustainability",
              "developer productivity tools", "code formatting debates"],
    "concept": ["why software projects often go over schedule",
                "why simple things in software are often hard",
                "why developers argue about tabs vs spaces"],
    "debate": ["whether AI will replace programmers",
               "monorepos vs polyrepos", "static vs dynamic typing",
               "whether code reviews slow teams down"],
    "behavior": ["underestimate how long coding tasks take",
                 "resist adopting new tools", "prefer building over buying",
                 "bikeshed on trivial decisions"],
    "thing": ["debugging", "reading other people's code", "system design",
              "developer tooling", "programming language design"],
    "subject": ["the future of programming", "work-life balance in tech",
                "the role of AI assistants in coding", "technical writing"],
    "hypothetical": ["start a tech company tomorrow", "learn programming from scratch",
                     "design a new programming language", "fix one thing about the internet"],
}

ARGUMENTATION = [
    "Make the case for {position}.",
    "Defend the claim that {claim}.",
    "Argue against {opposing_view}.",
    "Why is {opinion} wrong?",
    "Present both sides of {debate}, then take a position.",
    "Steelman the argument for {unpopular_position}.",
    "What's the strongest argument against {popular_position}?",
    "Convince a skeptic that {claim}.",
]

ARGUMENTATION_FILLS = {
    "position": ["using boring technology", "writing more tests",
                 "simplicity over features", "documentation as a first-class concern",
                 "investing in developer experience", "saying no to features"],
    "claim": ["most code should be deleted", "technical interviews are broken",
              "premature abstraction is worse than duplication",
              "most microservices should be monoliths",
              "comments are a code smell"],
    "opposing_view": ["move fast and break things", "always use the latest technology",
                      "code coverage targets", "mandatory code review for all changes",
                      "estimating in story points"],
    "opinion": ["rewriting from scratch is always bad",
                "you need microservices to scale",
                "100% test coverage is the goal",
                "senior developers should always lead projects"],
    "debate": ["build vs buy", "specialization vs generalization in engineering",
               "startups vs big companies for career growth",
               "formal CS education vs self-taught"],
    "unpopular_position": ["YAGNI taken to the extreme", "not writing tests for prototypes",
                           "copy-paste over abstraction", "premature optimization sometimes"],
    "popular_position": ["clean code principles", "test-driven development",
                         "microservices architecture", "agile methodology"],
}

# Length and formality variations
LENGTH_MODIFIERS = [
    ("", "medium"),
    ("Briefly ", "short"),
    ("In detail, ", "long"),
    ("In a sentence or two, ", "short"),
    ("Thoroughly ", "long"),
    ("Concisely ", "short"),
    ("Give me a comprehensive explanation of ", "long"),
]

FORMALITY_MODIFIERS = [
    ("", "neutral"),
    ("(Be casual and conversational) ", "casual"),
    ("(Write formally and precisely) ", "formal"),
    ("(Write like you're explaining to a colleague) ", "neutral"),
    ("(Keep it informal) ", "casual"),
    ("(Use technical language appropriately) ", "formal"),
]


def fill_template(template: str, fills: dict) -> str:
    """Fill a template with random values from the fills dict."""
    result = template
    for key, values in fills.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_prompts_for_category(
    templates: list[str],
    fills: dict,
    category: str,
    count: int,
    professional_weight: float = 1.0
) -> Iterator[dict]:
    """Generate prompts for a category with length/formality variations."""
    generated = 0
    attempts = 0
    max_attempts = count * 10  # Prevent infinite loops
    seen_prompts = set()

    while generated < count and attempts < max_attempts:
        attempts += 1
        template = random.choice(templates)
        base_prompt = fill_template(template, fills)

        # Skip duplicates
        if base_prompt in seen_prompts:
            continue
        seen_prompts.add(base_prompt)

        # Add length modifier
        length_mod, expected_length = random.choice(LENGTH_MODIFIERS)

        # Add formality modifier (weight toward neutral/formal for professional)
        if professional_weight > 0.5 and random.random() < professional_weight:
            formality_mod, formality = random.choice([
                ("", "neutral"),
                ("(Write formally and precisely) ", "formal"),
                ("(Write like you're explaining to a colleague) ", "neutral"),
            ])
        else:
            formality_mod, formality = random.choice(FORMALITY_MODIFIERS)

        prompt = f"{length_mod}{formality_mod}{base_prompt}".strip()

        yield {
            "category": category,
            "prompt": prompt,
            "expected_length": expected_length,
            "formality": formality,
        }
        generated += 1


def generate_all_prompts(total_target: int = 300) -> list[dict]:
    """
    Generate a diverse bank of prompts.

    Weighted toward technical/professional content:
    - Technical explanations: 30%
    - Analysis/opinion: 25%
    - Instruction/how-to: 25%
    - Creative writing: 10%
    - Casual conversation: 5%
    - Argumentation: 5%
    """
    prompts = []

    # Category distribution (weighted toward technical/professional)
    categories = [
        (TECHNICAL_EXPLANATIONS, TECHNICAL_FILLS, "technical", 0.30),
        (ANALYSIS_OPINION, ANALYSIS_FILLS, "analysis", 0.25),
        (INSTRUCTION_HOWTO, INSTRUCTION_FILLS, "instruction", 0.25),
        (CREATIVE_WRITING, CREATIVE_FILLS, "creative", 0.10),
        (CASUAL_CONVERSATION, CASUAL_FILLS, "casual", 0.05),
        (ARGUMENTATION, ARGUMENTATION_FILLS, "argumentation", 0.05),
    ]

    for templates, fills, category, weight in categories:
        count = int(total_target * weight)
        prompts.extend(generate_prompts_for_category(
            templates, fills, category, count,
            professional_weight=0.8 if category in ["technical", "analysis", "instruction"] else 0.3
        ))

    # Shuffle to mix categories
    random.shuffle(prompts)

    # Add IDs
    for i, prompt in enumerate(prompts):
        prompt["id"] = f"prompt_{i:04d}"

    return prompts


def save_prompts(prompts: list[dict], output_path: Path) -> None:
    """Save prompts to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")


def main(output_path: Path, num_prompts: int = 300) -> int:
    """Generate and save prompts."""
    print(f"Generating {num_prompts} prompts...")
    prompts = generate_all_prompts(num_prompts)
    save_prompts(prompts, output_path)
    print(f"Saved {len(prompts)} prompts to {output_path}")

    # Print category distribution
    from collections import Counter
    cats = Counter(p["category"] for p in prompts)
    print("\nCategory distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(prompts)*100:.1f}%)")

    return len(prompts)


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "prompts.jsonl"
    main(output_path)
