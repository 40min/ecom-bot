# E-commerce Bot Demo

A demonstration CLI application showcasing LangChain agents with tool integration and style evaluation for customer service automation.

## Purpose

This is a **demo/test project** created to explore and experiment with LangChain's agent capabilities. It demonstrates building a customer service bot with configurable personalities and automated style evaluation.

## Features

- **CLI-based customer service bot** for e-commerce inquiries
- **LangChain agent integration** with custom tools
- **Configurable bot personalities** (roles) via YAML configuration
- **Style evaluation mode** with rule-based and LLM-based scoring
- **Order lookup functionality** via database simulation
- **FAQ-based responses** using structured data
- **Session management** with conversation memory
- **Structured logging** with JSON formatting
- **Russian language support** for customer interactions

## Architecture

- [`app_lc.py`](app_lc.py:1) - Main CLI entry point with bot and evaluate commands
- [`src/bot.py`](src/bot.py:1) - LangChain-based CLI bot implementation
- [`src/style_eval.py`](src/style_eval.py:1) - Style evaluation system with async batch processing
- [`src/prompts/style_config.py`](src/prompts/style_config.py:1) - Bot personality configuration loader
- [`src/orders_db.py`](src/orders_db.py:1) - Order database simulation and lookup tools
- [`data/style_guide.yaml`](data/style_guide.yaml:1) - Bot personality definitions
- [`data/faq.json`](data/faq.json:1) - Customer service FAQ data
- [`data/orders.json`](data/orders.json:1) - Sample order data for testing
- [`data/eval_prompts.txt`](data/eval_prompts.txt:1) - Test prompts for evaluation mode

## Tech Stack

- **LangChain** - Agent framework and LLM integration
- **OpenRouter/OpenAI API** - Language model access (configurable)
- **Python 3.13+** - Runtime environment
- **Pydantic** - Data validation and structured outputs
- **PyYAML** - Configuration management
- **In-memory storage** - Session and data management

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API key and settings
   ```
   
   **Supported LLM providers:**
   - **OpenRouter** (default): `API_URL=https://openrouter.ai/api/v1`
   - **OpenAI**: `API_URL=https://api.openai.com/v1`

3. **Run the bot:**
   ```bash
   python app_lc.py bot
   # or use: make run-bot
   ```

## Usage

### Main Commands

**Interactive Bot Mode:**
```bash
python app_lc.py bot
# or: make run-bot
```

**Evaluation Mode:**
```bash
uv run python app_lc.py evaluate-style
# or: make run-style-eval

# With custom evaluation model:
uv run python app_lc.py evaluate-style --eval-model gpt-4o
# or: make run-style-eval-custom MODEL=gpt-4o
```

### Supplemental Commands

```bash
make help              # Show all available commands
make dev-setup         # Complete development setup
make check-env         # Verify environment configuration
make status            # Show project status
make clean             # Clean temporary files
```

## Bot Personalities (Roles)

Bot personalities are configured in [`data/style_guide.yaml`](data/style_guide.yaml:1). Each role defines:

- **Character name and description** - Personality traits and tone
- **Avoid list** - Elements to exclude (emojis, excessive punctuation, etc.)
- **Must include** - Required elements (simple language, solution options, etc.)
- **Fallback responses** - Default answers when data is unavailable

**Available roles:**
- **alex** - Polite, business-friendly tone without being overly casual
- **pahom** - Old Russian dialect with folksy expressions and diminutives

**Setting the role:**
```bash
# In .env file:
PERSON_NAME=alex  # or pahom
```

The bot automatically loads the corresponding personality configuration and few-shot examples from [`data/few_shots_{person}.jsonl`](data/few_shots_alex.jsonl:1).

## Evaluation Mode

Evaluation mode tests bot responses against style guidelines using a hybrid scoring system:

**Scoring Components:**
- **Rule-based checks (40%)** - Automated detection of style violations:
  - Emoji presence
  - Excessive exclamation marks
  - Response length limits
- **LLM-based grading (60%)** - AI evaluation of tone and style adherence

**Setup:**

1. Create test prompts in [`data/eval_prompts.txt`](data/eval_prompts.txt:1) (one per line)
2. Configure evaluation model in command or use default
3. Run evaluation:
   ```bash
   python app_lc.py evaluate --eval-model gpt-4o-mini
   ```

**Output:**
- Detailed report: [`reports/style_eval.json`](reports/style_eval.json:1)
- Summary: [`reports/style_eval_summary.json`](reports/style_eval_summary.json:1)
- Console output with statistics and pass/fail rates

The evaluator uses async batch processing with rate limiting for efficient testing of multiple prompts.

## Interactive Bot Examples

Once running in bot mode, try these interactions:

- Ask general questions: *"Сколько идёт доставка?"*
- Check order status: *"/order 12345"* or *"Мой заказ 12345"*
- Reset conversation: *"сброс"*
- Exit: *"выход"*, *"стоп"*, or *"конец"*

## Key Learning Points

- **LangChain Agents**: Building AI agents with tool access
- **Tool Integration**: Creating custom tools for specific tasks
- **Personality Configuration**: YAML-based bot personality management
- **Style Evaluation**: Hybrid rule-based and LLM-based quality assessment
- **Memory Management**: Conversation state and session handling
- **Async Processing**: Efficient batch evaluation with rate limiting
- **Error Handling**: Robust API interaction patterns
- **Logging**: Structured logging for debugging and monitoring

## Demo Limitations

This is a **simplified demonstration** with:
- In-memory data storage (no persistent database)
- Mock order database for testing
- Basic error handling
- CLI-only interface (no web interface)
- Russian language focus
- Limited to configured personalities

## Development

The project uses:
- `langchain` and `langchain-openai` for LLM integration
- `python-dotenv` for environment configuration
- `pydantic` for data validation and structured outputs
- `pyyaml` for configuration management
- `click` for CLI interface
- `emoji` for emoji detection in evaluation

Perfect for learning LangChain concepts, agent patterns, personality configuration, and automated quality evaluation in a controlled environment.

---

**Note**: This is a demonstration project for educational purposes. Not intended for production use.
