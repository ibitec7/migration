"""
Prompt templates for news article summarization using FLAN-T5.

This module provides configurable prompt templates designed to extract
key events, important information, and main points from migration-related
news articles.
"""



# Primary summarization template
SUMMARIZATION_PROMPT = """Summarize the following news article focusing on extracting key events, important facts, and main impacts. Provide a concise summary that highlights the most critical information about migration policies, enforcement actions, and related developments.

Article:
{article_text}

Summary:"""


# Alternative templates for different summarization styles
EXTRACTION_PROMPT = """Extract the key events and important information from this migration news article. Focus on:
- Main policy changes or announcements
- Enforcement actions and statistics
- Dates, locations, and affected populations
- Government or organizational responses

Article:
{article_text}

Key Information:"""


EVENTS_FOCUSED_PROMPT = """What are the main events described in this migration news article? Provide a structured summary of:
1. Primary event(s)
2. Key stakeholders involved
3. Impact or implications

Article:
{article_text}

Events Summary:"""


class PromptTemplate:
    """Configurable prompt template for news summarization."""
    
    def __init__(
        self,
        template: str,
        max_input_length: int = 1024,
        max_output_length: int = 256,
        style: str = "extractive"
    ):
        """
        Initialize prompt template.
        
        Args:
            template: Template string with {article_text} placeholder
            max_input_length: Maximum input article length in tokens (approx)
            max_output_length: Maximum summary length in tokens
            style: Summarization style ('extractive', 'abstractive', 'events')
        """
        self.template = template
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.style = style
    
    def format(self, article_text: str) -> str:
        """
        Format the prompt with article text.
        
        Args:
            article_text: The article content to summarize
            
        Returns:
            Formatted prompt string
        """
        # Truncate article if needed (rough estimate: 4 chars per token)
        max_chars = self.max_input_length * 4
        if len(article_text) > max_chars:
            article_text = article_text[:max_chars]
        
        return self.template.format(article_text=article_text.strip())
    
    def __call__(self, article_text: str) -> str:
        """Make template callable."""
        return self.format(article_text)


# Pre-configured templates
MIGRATION_NEWS_SUMMARIZER = PromptTemplate(
    template=SUMMARIZATION_PROMPT,
    max_input_length=1024,
    max_output_length=256,
    style="extractive"
)

MIGRATION_EVENT_EXTRACTOR = PromptTemplate(
    template=EXTRACTION_PROMPT,
    max_input_length=1024,
    max_output_length=256,
    style="extractive"
)

MIGRATION_EVENTS_FOCUSED = PromptTemplate(
    template=EVENTS_FOCUSED_PROMPT,
    max_input_length=1024,
    max_output_length=256,
    style="events"
)


def get_prompt_template(name: str = "default") -> PromptTemplate:
    """
    Get a pre-configured prompt template by name.
    
    Args:
        name: Template name ('default', 'extraction', 'events')
        
    Returns:
        PromptTemplate instance
        
    Raises:
        ValueError: If template name not found
    """
    templates = {
        "default": MIGRATION_NEWS_SUMMARIZER,
        "extraction": MIGRATION_EVENT_EXTRACTOR,
        "events": MIGRATION_EVENTS_FOCUSED,
    }
    
    if name not in templates:
        raise ValueError(
            f"Unknown template '{name}'. Available: {', '.join(templates.keys())}"
        )
    
    return templates[name]
