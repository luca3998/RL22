# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Install the conda environment by the following command. Updating the environment can be done by the second line.

```bash
conda env create --file conda_env.yml
conda env update --file conda_env.yml  --prune
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
