.PHONY: help install install-dev setup-env run-bot run-eval test lint format clean logs-setup dev-setup

# Default target
help:
	@echo "E-commerce Bot - Available Commands:"
	@echo ""
	@echo "=== Running the Application ==="
	@echo "  run-bot        Run the bot in interactive mode"
	@echo "  run-eval       Run the bot in evaluation mode"
	@echo "  run-eval-custom Run evaluation with custom model (specify MODEL)"
	@echo ""
	@echo "=== Development Setup ==="
	@echo "  install        Install project dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup-env      Set up environment variables file"
	@echo "  dev-setup      Complete development setup (install + dev + env)"
	@echo ""
	@echo "=== Development Tools ==="
	@echo "  test           Run tests"
	@echo "  lint           Run linting (ruff)"
	@echo "  lint-fix       Run linting and fix issues (ruff --fix)"
	@echo "  format         Format code (black)"
	@echo "  format-check   Check code formatting (black --check)"
	@echo ""
	@echo "=== Maintenance ==="
	@echo "  clean          Clean temporary files and logs"
	@echo "  logs-setup     Create logs directory"
	@echo ""
	@echo "=== Utility ==="
	@echo "  check-env      Check if environment is properly configured"

# =============================================================================
# APPLICATION RUNNING
# =============================================================================

# Run the bot in interactive mode
run-bot:
	@echo "🚀 Starting bot in interactive mode..."
	python app_lc.py bot

# Run the bot in evaluation mode
run-eval:
	@echo "📊 Starting bot in evaluation mode..."
	python app_lc.py evaluate

# Run evaluation with custom model
run-eval-custom:
	@echo "📊 Starting bot in evaluation mode with custom model..."
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: Please specify MODEL, e.g., make run-eval-custom MODEL=gpt-4o"; \
		exit 1; \
	fi
	python app_lc.py evaluate --eval-model $(MODEL)

# =============================================================================
# DEVELOPMENT SETUP
# =============================================================================

# Install project dependencies
install:
	@echo "📦 Installing project dependencies..."
	uv sync

# Install development dependencies
install-dev:
	@echo "📦 Installing development dependencies..."
	uv sync --extra dev

# Set up environment file
setup-env:
	@echo "⚙️ Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "✅ Created .env from .env.example"; \
			echo "⚠️  Please edit .env and add your API keys"; \
		else \
			echo "⚠️  No .env.example found. Creating basic .env file..."; \
			echo "OPENROUTER_API_KEY=your_api_key_here" > .env; \
			echo "OPENROUTER_API_MODEL=gpt-4o-mini" >> .env; \
			echo "PERSON_NAME=alex" >> .env; \
			echo "✅ Created basic .env file"; \
		fi \
	else \
		echo "ℹ️  .env file already exists"; \
	fi

# Complete development setup
dev-setup: install-dev setup-env logs-setup
	@echo "✅ Development setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file and add your API keys"
	@echo "2. Run 'make run-bot' to start the bot"
	@echo "3. Run 'make run-eval' to test evaluation mode"

# Create necessary directories
logs-setup:
	@echo "📁 Creating necessary directories..."
	@mkdir -p logs
	@mkdir -p reports
	@echo "✅ Directories created"

# =============================================================================
# DEVELOPMENT TOOLS
# =============================================================================

# Run tests
test:
	@echo "🧪 Running tests..."
	uv run pytest -v

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	uv run pytest --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	@echo "🔍 Running linting..."
	uv run ruff check .
	@echo "✅ Linting complete"

# Lint and fix issues
lint-fix:
	@echo "🔧 Fixing linting issues..."
	uv run ruff check --fix .
	@echo "✅ Linting fixes applied"

# Format code
format:
	@echo "✨ Formatting code..."
	uv run black .
	@echo "✅ Code formatting complete"

# Check code formatting
format-check:
	@echo "🔍 Checking code formatting..."
	uv run black --check .
	@echo "✅ Format check complete"

# Run both linting and formatting
format-all: format lint
	@echo "✅ Code formatting and linting complete"

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

# Check environment configuration
check-env:
	@echo "🔍 Checking environment configuration..."
	@if [ ! -f .env ]; then \
		echo "❌ .env file not found. Run 'make setup-env' to create it."; \
		exit 1; \
	fi
	@echo "✅ .env file exists"
	@if grep -q "OPENROUTER_API_KEY=your_api_key_here" .env; then \
		echo "⚠️  API key not configured. Please edit .env file."; \
	else \
		echo "✅ API key appears to be configured"; \
	fi
	@if [ ! -d "logs" ]; then \
		echo "⚠️  Logs directory not found. Run 'make logs-setup' to create it."; \
	else \
		echo "✅ Logs directory exists"; \
	fi

# Show project status
status:
	@echo "📊 Project Status:"
	@echo "Python version: $$(python --version)"
	@echo "uv version: $$(uv --version 2>/dev/null || echo 'uv not installed')"
	@if [ -f .env ]; then echo "✅ .env file exists"; else echo "❌ .env file missing"; fi
	@if [ -d logs ]; then echo "✅ logs directory exists"; else echo "❌ logs directory missing"; fi
	@if [ -d reports ]; then echo "✅ reports directory exists"; else echo "❌ reports directory missing"; fi
	@echo "Environment variables:"
	@grep -E "^(OPENROUTER_|PERSON_NAME)" .env 2>/dev/null || echo "No environment variables found"

# =============================================================================
# MAINTENANCE
# =============================================================================

# Clean temporary files and logs
clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@echo "✅ Temporary files cleaned"

# Clean logs (keep directory structure)
clean-logs:
	@echo "🧹 Cleaning log files..."
	@find logs -name "*.jsonl" -delete 2>/dev/null || true
	@find reports -name "*.json" -delete 2>/dev/null || true
	@echo "✅ Log files cleaned"

# Full clean (including dependencies)
clean-all: clean clean-logs
	@echo "🧹 Full clean complete"
	@echo "Run 'make install' to reinstall dependencies"

# =============================================================================
# DEPLOYMENT HELPERS
# =============================================================================

# Check if ready for deployment
check-deploy:
	@echo "🔍 Checking deployment readiness..."
	@make check-env
	@make format-check
	@make lint
	@make test
	@echo "✅ All checks passed - ready for deployment!"

# Show useful information
info:
	@echo "📖 Project Information:"
	@echo "Name: $$(grep '^name = ' pyproject.toml | cut -d'"' -f2)"
	@echo "Version: $$(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
	@echo "Python: $$(grep 'requires-python' pyproject.toml | cut -d'=' -f2 | tr -d ' \"')"
	@echo ""
	@echo "📁 Key Files:"
	@echo "  - app_lc.py: Main application entry point"
	@echo "  - src/bot.py: Bot implementation"
	@echo "  - src/style_eval.py: Evaluation system"
	@echo "  - data/: Data files (FAQ, examples, etc.)"
	@echo ""
	@echo "🔧 Configuration:"
	@echo "  - .env: Environment variables (copy from .env.example)"
	@echo "  - logs/: Session logs"
	@echo "  - reports/: Evaluation reports"