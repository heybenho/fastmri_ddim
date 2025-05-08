import yaml
def load_config_to_args(parser, config_path, config_section=None):
    """
    Load configuration from YAML file and set defaults on the parser.
    Optionally filter by a specific section in the config file.
    
    Args:
        parser (ArgumentParser): The parser to update
        config_path (str or Path): Path to the config file
        section (str, optional): Specific section of the config to load
    
    Returns:
        ArgumentParser: The updated parser
    """
    # try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    if config_section and config_section in config:
        config = config[config_section]

    for key, value in config.items():
        if not isinstance(value, dict):
            print(f"Setting default for {key}: {value}")
            # parser.set_defaults(**{key: value})
                
    #     return parser
    # except FileNotFoundError:
    #     print(f"Warning: Configuration file {config_path} not found")
    #     return parser
    # except yaml.YAMLError as e:
    #     print(f"Error parsing YAML file: {e}")
    #     return parser

# Example usage:
# In build_args(), you can use this to load different sections for different parsers:
# parser = load_config_to_args(parser, "ddim_config.yaml", section="data")
# parser = DDIMModule.add_model_specific_args(parser)
# parser = load_config_to_args(parser, "ddim_config.yaml", section="model")

parser = ""
load_config_to_args(parser, "ddim_config.yaml", config_section="training")