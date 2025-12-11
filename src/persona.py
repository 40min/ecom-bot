"""
Persona module for loading and managing bot personality configurations.
"""

import yaml
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PersonaDetails(BaseModel):
    """Pydantic model for individual persona configuration."""
    name: str = Field(..., description="Character name")
    persona: str = Field(..., description="Character description")
    avoid: List[str] = Field(default_factory=list, description="Things to avoid in responses")
    must_include: List[str] = Field(default_factory=list, description="Required elements in responses")
    fallback: Dict[str, str] = Field(default_factory=dict, description="Fallback responses for edge cases")


class ToneConfig(BaseModel):
    """Pydantic model for tone configuration."""
    persons: Dict[str, PersonaDetails] = Field(..., description="All available personas")
    sentences_max: int = Field(default=3, description="Maximum sentences per response")
    bullets: bool = Field(default=True, description="Whether to use bullet points")


class StyleGuide(BaseModel):
    """Root model for the complete style guide configuration."""
    brand: str = Field(..., description="Brand name")
    tone: ToneConfig = Field(..., description="Tone configuration")    


class Persona:
    """Main interface class for persona management."""
    
    def __init__(self, config: StyleGuide, persona_name: str):
        """
        Initialize Persona with loaded configuration.
        
        Args:
            config: Validated StyleGuide configuration
            persona_name: Name of the persona to use (e.g., 'alex', 'pahom')
        """
        self.config = config
        self.persona_name = persona_name
        
        # Validate persona exists
        if persona_name not in config.tone.persons:
            available_personas = list(config.tone.persons.keys())
            raise ValueError(
                f"Persona '{persona_name}' not found. "
                f"Available personas: {available_personas}"
            )
        
        self.persona_details = config.tone.persons[persona_name]
    
    @classmethod
    def load_persona(cls, persona_name: str, yaml_path: str = './data/style_guide.yaml') -> 'Persona':
        """
        Load persona configuration from YAML file.
        
        Args:
            persona_name: Name of the persona to load (e.g., 'alex', 'pahom')
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Persona: Configured Persona object
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file has invalid format
            ValueError: If persona_name is not found in configuration
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                yaml_data = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML format in {yaml_path}: {e}")
        
        # Validate and parse with Pydantic
        try:
            config = StyleGuide(**yaml_data)
        except Exception as e:
            raise ValueError(f"Invalid configuration format: {e}")
        
        return cls(config, persona_name)
    
    def get_system_prompt_addition(self) -> str:
        """
        Generate system prompt section with persona details.
        
        Returns:
            str: Formatted system prompt addition with persona rules
        """
        persona = self.persona_details
        tone_config = self.config.tone        
        
        # Build avoid list
        avoid_list = "\n".join([f"  - {item}" for item in persona.avoid])
        
        # Build must_include list
        must_include_list = "\n".join([f"  - {item}" for item in persona.must_include])
        
        # Build fallback response
        fallback_response = persona.fallback.get('no_data', 'Извините, у меня нет информации по этому вопросу.')        
        
        prompt_addition = f"""
Персона: {persona.persona}

Правила общения:
Избегай:
{avoid_list}

Обязательно используй:
{must_include_list}

Ограничения:
- Максимум {tone_config.sentences_max} предложений в ответе
- {"Используй списки и маркированные пункты" if tone_config.bullets else "Избегай использования списков"}

При отсутствии данных: {fallback_response}

"""
        
        return prompt_addition.strip()
    
    @property
    def brand(self) -> str:
        """Get the brand name from configuration."""
        return self.config.brand
    
    @property
    def available_personas(self) -> List[str]:
        """Get list of all available persona names."""
        return list(self.config.tone.persons.keys())
    
    def __str__(self) -> str:
        """String representation of the persona."""
        return f"Persona(name={self.persona_name}, brand={self.brand})"
    
    def __repr__(self) -> str:
        """Detailed representation of the persona."""
        return f"Persona(name={self.persona_name}, brand={self.brand}, available={self.available_personas})"
