# Contribution guideline

This document describes the guidelines for contributing to the project. Refer to this document when in doubt.

- [Contribution guideline](#contribution-guideline)
  - [Adding code to the repository](#adding-code-to-the-repository)
  - [Dependencies](#dependencies)

## Adding code to the repository

1. Is there a feature that already exists / is very closed to what I want to do ? If it does :
   - Maybe the feature is enough
   - Look into refactoring the feature first
2. Is there an existing file in which the feature would be relevant ? If it does :
   - Add a function within this file
   - If the file becomes too large, [see this section on refactoring](/documentation/project_refactoring.md#how-to-refactor) 
3. Is there an existing directory in which I can create the file ? If it does :
   - Create the file by following the [naming convention](/documentation/best_practices.md#modules)
4. If you need to create a new directory, please talk about it to the others as you might have missed something in the first three steps.

## Dependencies

Dependencies are added to the [pyproject.toml file](/pyproject.toml). When adding a dependency, two questions are to be awnsered :
- Can I use a native feature ? Is it worth implementing the code myself
- If not, is this a production dependency ?

If this is a production dependency, add it to the [dependencies list](/pyproject.toml#L22). Else, it can either be a [dev dependency](/pyproject.toml#L29), a [test dependency](/pyproject.toml#L33), or a custom group.  
To create a custom group, look at [pyproject.toml group system](https://packaging.python.org/en/latest/specifications/dependency-groups/).

