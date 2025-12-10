# E-commerce Bot Demo

A demonstration CLI application showcasing LangChain agents with tool integration for customer service automation.

## Purpose

This is a **demo/test project** created to explore and experiment with LangChain's agent capabilities. It's designed as a learning example showing how to build a simple customer service bot using modern LLM frameworks.

## Features

- **CLI-based customer service bot** for e-commerce inquiries
- **LangChain agent integration** with custom tools
- **Order lookup functionality** via database simulation
- **FAQ-based responses** using structured data
- **Session management** with conversation memory
- **Structured logging** with JSON formatting
- **Russian language support** for customer interactions

## Architecture

- `app_lc.py` - Main application entry point with logging setup
- `src/bot.py` - LangChain-based CLI bot implementation
- `src/order_db.py` - Order database simulation and lookup tools
- `data/faq.json` - Customer service FAQ data
- `data/orders.json` - Sample order data for testing

## Tech Stack

- **LangChain** - Agent framework and LLM integration
- **OpenAI API** - Language model access
- **Python 3.13+** - Runtime environment
- **In-memory storage** - Session and data management

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Run the demo:**
   ```bash
   python app_lc.py
   ```

## Usage Examples

Once running, try these interactions:

- Ask general questions: *"Сколько идёт доставка?"*
- Check order status: *"/order 12345"* or *"Мой заказ 12345"*
- Reset conversation: *"сброс"*
- Exit: *"выход"*, *"стоп"*, or *"конец"*

## Key Learning Points

- **LangChain Agents**: Building AI agents with tool access
- **Tool Integration**: Creating custom tools for specific tasks
- **Memory Management**: Conversation state and session handling
- **Error Handling**: Robust API interaction patterns
- **Logging**: Structured logging for debugging and monitoring

## Demo Limitations

This is a **simplified demonstration** with:
- In-memory data storage (no persistent database)
- Mock order database for testing
- Basic error handling
- CLI-only interface (no web interface)
- Russian language focus

## Development

The project uses:
- `langchain` and `langchain-openai` for LLM integration
- `python-dotenv` for environment configuration
- `pydantic` for data validation

Perfect for learning LangChain concepts, agent patterns, and tool integration in a controlled environment.

---

**Note**: This is a demonstration project for educational purposes. Not intended for production use.