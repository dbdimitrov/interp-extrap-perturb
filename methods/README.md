# Methods Directory

This directory contains individual YAML files for each computational method in the catalog. Each file represents one method with its complete metadata and description.

## File Structure

Each method file follows this structure:

```yaml
Code Availability: https://github.com/example/method
Description: |
  A detailed description of the method, including its approach,
  key features, and how it works.
Inspired by:
  - Reference 1
  - Reference 2
Method: Method Name
Model:
  - Model Type 1
  - Model Type 2
Publication: https://doi.org/10.1000/example
Published: true
Task:
  - Task Category 1
  - Task Category 2
Year: 2024
```

## Adding a New Method

1. Create a new YAML file with a descriptive filename (e.g., `my_method.yaml`)
2. Use lowercase letters, numbers, hyphens, and underscores only in the filename
3. Follow the structure above, ensuring all required fields are included
4. The filename will be automatically used to identify the method

## Required Fields

- **Method**: The name of the method
- **Year**: Publication year
- **Description**: Detailed description of the method
- **Publication**: Link to the publication or DOI
- **Code Availability**: Link to code repository, or '-' if not available
- **Published**: Boolean (true/false) indicating if formally published
- **Task**: List of tasks/applications the method addresses

## Optional Fields

- **Model**: List of underlying models or frameworks
- **Inspired by**: List of references or inspirations

## File Naming

- Use descriptive names based on the method name
- Convert to lowercase and replace special characters with underscores
- Examples:
  - `Method: cPCA` → `cpca.yaml`
  - `Method: GEARS` → `gears.yaml`
  - `Method: scGEN` → `scgen.yaml`

