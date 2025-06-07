# Publishing SpinLab to PyPI

## Prerequisites

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Create TestPyPI account**: https://test.pypi.org/account/register/
3. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Step-by-Step Publishing

### 1. Prepare the Package

```bash
cd /Users/akramibrahim/SpinLab

# Clean previous builds
rm -rf dist/ build/ *.egg-info/
```

### 2. Build the Package

```bash
python -m build
```

This creates:
- `dist/spinlab_sim-0.1.0.tar.gz` (source distribution)
- `dist/spinlab_sim-0.1.0-py3-none-any.whl` (wheel)

### 3. Test on TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ spinlab-sim
```

### 4. Upload to Real PyPI

```bash
twine upload dist/*
```

### 5. Test Final Installation

```bash
pip install spinlab-sim
```

## After Publishing

Users can install with:
```bash
pip install spinlab-sim
```

## Version Updates

To release new versions:

1. Update version in `pyproject.toml`
2. Clean and rebuild: `rm -rf dist/ && python -m build`
3. Upload: `twine upload dist/*`

## Package Name Note

- Package name on PyPI: `spinlab-sim`
- Import name in Python: `spinlab`
- This avoids conflicts with existing packages

## Authentication

For automated publishing, set up API tokens:
1. Go to PyPI → Account Settings → API tokens
2. Create token with scope for your project
3. Use token instead of username/password