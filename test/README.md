# Unit Test (UT) Guide

This project uses `pytest` for unit testing. All test cases are under the `test/` directory. Below is a simple guide for new users to write and run UTs:

## 1. Environment Setup
- Recommended Python 3.8 or above.
- Install dependencies:
  ```sh
  pip install -r ../requirements.txt
  pip install pytest
  ```

## 2. Test Structure
- Place your test files in the `test/` directory, and name them starting with `test_`.
- You can refer to existing `test_*.py` files.
- Common fixtures (such as `tiny_opt_model`, `opt_model`, `opt_tokenizer`, `dataloader`) and helper functions (such as `model_infer`) are defined in `confest.py` and `helpers.py` and can be imported directly.
- Example:
  ```python
  # test_example.py
    from ..helpers import model_infer

    def test_model_infer(tiny_opt_model, opt_tokenizer):
        result = model_infer(tiny_opt_model, opt_tokenizer, input_text="hello world")
        assert result is not None
  ```

## 3. Running Tests
- In the `test/` directory, run:
  ```sh
  pytest
  ```
- You can specify a single file or test case:
  ```sh
  pytest test_xxx.py
  pytest -k "test_func_name"
  ```

## 4. Debugging Tips
- `confest.py` adds the parent directory to `sys.path`, so you can debug without installing the local package.
- You can directly import project source code in your test cases.

## 5. Reference
- Fixtures are defined in `confest.py` and `fixtures.py`
- Helper functions are in `helpers.py`

If you have any questions, feel free to open an issue.
