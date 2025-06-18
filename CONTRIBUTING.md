# Contributing to Mental Health Data Analysis Dashboard

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button at the top right of the repository page
- Clone your fork locally:
```bash
git clone https://github.com/yourusername/mental-health-dashboard.git
cd mental-health-dashboard
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if any)
pip install -r requirements-dev.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/amazing-feature
# or
git checkout -b fix/bug-description
```

## 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report:

1. **Use a clear title** that describes the problem
2. **Describe the exact steps** to reproduce the issue
3. **Include screenshots** if applicable
4. **Specify your environment**: OS, Python version, browser
5. **Include error messages** and stack traces

## 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please:

1. **Check existing feature requests** first
2. **Explain the problem** the enhancement would solve
3. **Describe the solution** you'd like to see
4. **Consider alternatives** you've thought about

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and small

### Testing
- Add tests for new features
- Ensure existing tests pass
- Test with different data formats
- Verify UI components work correctly

### Documentation
- Update README.md if needed
- Add docstrings to new functions
- Update inline comments
- Consider adding examples

## 📋 Pull Request Process

1. **Update documentation** as needed
2. **Add tests** for new functionality
3. **Ensure all tests pass**
4. **Update the README** with details of changes if applicable
5. **Follow the pull request template**

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots of UI changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

## 🏷️ Issue Labels

We use these labels to categorize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `priority: high` - High priority issues

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations
- Better error handling
- Mobile responsiveness
- Accessibility improvements

### Medium Priority
- Additional chart types
- Export format options
- Theme customization
- Data validation

### Low Priority
- UI enhancements
- Code refactoring
- Documentation improvements
- Example datasets

## 📦 Project Structure

```
mental-health-dashboard/
├── streamlit_app.py          # Main application
├── requirements.txt          # Dependencies
├── README.md                # Project documentation
├── LICENSE                  # License file
├── .gitignore              # Git ignore rules
├── SECURITY.md             # Security policy
├── CONTRIBUTING.md         # This file
├── docs/                   # Additional documentation
├── tests/                  # Test files
├── assets/                 # Images, icons, etc.
└── examples/               # Example datasets
```

## 🔍 Code Review Process

1. **Automated checks** must pass
2. **At least one maintainer** will review
3. **Changes requested** may need addressing
4. **Final approval** required before merge

## 🎉 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Credited in the application (optional)

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: [your-email@example.com] for sensitive matters

## 📜 Code of Conduct

This project follows a code of conduct. Please be respectful and constructive in all interactions.

---

Thank you for contributing to mental health data analysis tools! 🧠💙
