"""
Style configuration module for loading and managing bot personality configurations.
"""

import yaml
from typing import Dict, List
from pydantic import BaseModel, Field


class PersonDetails(BaseModel):
    """Pydantic model for individual person configuration."""
    name: str = Field(..., description="Character name")
    person: str = Field(..., description="Character description")
    avoid: List[str] = Field(default_factory=list, description="Things to avoid in responses")
    must_include: List[str] = Field(default_factory=list, description="Required elements in responses")
    fallback: Dict[str, str] = Field(default_factory=dict, description="Fallback responses for edge cases")


class ToneConfig(BaseModel):
    """Pydantic model for tone configuration."""
    persons: Dict[str, PersonDetails] = Field(..., description="All available personas")
    sentences_max: int = Field(default=3, description="Maximum sentences per response")
    bullets: bool = Field(default=True, description="Whether to use bullet points")


class StyleGuide(BaseModel):
    """Root model for the complete style guide configuration."""
    brand: str = Field(..., description="Brand name")
    tone: ToneConfig = Field(..., description="Tone configuration")    


class StyleConfig:
    """Main interface class for person management."""
    
    def __init__(self, config: StyleGuide, person_name: str):
        """
        Initialize Person with loaded configuration.
        
        Args:
            config: Validated StyleGuide configuration
            persona_name: Name of the person to use (e.g., 'alex', 'pahom')
        """
        self.config = config
        self.person_name = person_name        
    
    @classmethod
    def load(cls, person_name: str, yaml_path: str = './data/style_guide.yaml') -> 'StyleConfig':
        """
        Load style configuration from YAML file.
        
        Args:
            person_name: Name of the person to load (e.g., 'alex', 'pahom')
            yaml_path: Path to the YAML configuration file
            
        Returns:
            StyleConfig: Configured StyleConfig object
            
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
        
        # Validate person exists
        if person_name not in config.tone.persons:
            available_personas = list(config.tone.persons.keys())
            raise ValueError(
                f"Person '{person_name}' not found. "
                f"Available personas: {available_personas}"
            )
        
        return cls(config, person_name)
    
    def get_system_prompt_addition(self) -> str:
        """
        Generate system prompt section with person details.
        
        Returns:
            str: Formatted system prompt addition with person rules
        """
        person = self.config.tone.persons[self.person_name]
        tone_config = self.config.tone        
        
        # Build avoid list
        avoid_list = "\n".join([f"  - {item}" for item in person.avoid])
        
        # Build must_include list
        must_include_list = "\n".join([f"  - {item}" for item in person.must_include])
        
        # Build fallback response
        fallback_response = person.fallback.get('no_data', 'Извините, у меня нет информации по этому вопросу.')        
        
        prompt_addition = f"""
Ты {person.name} полезный сотрудник интернет-магазина {self.brand}.
Характер: {person.person}

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
    def available_persons(self) -> List[str]:
        """Get list of all available person names."""
        return list(self.config.tone.persons.keys())
    
    @property
    def current_person_description(self) -> str:
        """Get current person tone from configuration."""
        person = self.config.tone.persons[self.person_name]
        return person.person
    
    @property
    def current_person_avoid(self) -> List[str]:
        """Get current person avoid list from configuration."""

        person = self.config.tone.persons[self.person_name]
        return person.avoid
    
    @property
    def current_person_must_include(self) -> List[str]:
        """Get current person must include list from configuration."""
        person = self.config.tone.persons[self.person_name]
        return person.must_include
