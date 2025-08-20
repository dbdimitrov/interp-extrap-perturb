# 🧪 Methods Directory

<div align="center">

*Individual YAML files for each computational method in the catalog*

![Methods Count](https://img.shields.io/badge/methods-143-blue.svg)
![Format](https://img.shields.io/badge/format-YAML-green.svg)
![Structure](https://img.shields.io/badge/structure-standardized-orange.svg)

</div>

---

## 📋 **Overview**

This directory contains **individual YAML files** for each computational method in our catalog. Each file represents one method with its complete metadata and description, making it easy for contributors to add new methods or edit existing ones.

> 💡 **Why individual files?** This structure makes contributions easier, reduces merge conflicts, and allows for better organization and maintenance.

---

## 🏗️ **File Structure**

Each method file follows this standardized structure:

```yaml
# 📝 Basic Information
Method: Method Name                    # The official method name
Year: 2024                            # Publication year
Published: true                        # Publication status (true/false)

# 🔬 Scientific Details  
Description: |
  A detailed description of the method, including its approach,
  key features, and how it works. This appears in the expandable
  table rows on the website.

# 🏷️ Classification
Task:                                  # List of computational tasks
  - Task Category 1
  - Task Category 2
Model:                                 # Underlying models/frameworks  
  - Model Type 1
  - Model Type 2

# 🔗 References & Code
Publication: https://doi.org/10.1000/example    # DOI or publication URL
Code Availability: https://github.com/example/method  # Code repository URL
Inspired by:                           # Related methods/papers
  - Reference 1
  - Reference 2
```

---

## ➕ **Adding a New Method**

### 🚀 **Quick Start**

1. **📁 Create File** — New YAML file with descriptive filename (e.g., `my_awesome_method.yaml`)
2. **📝 Use Template** — Copy the structure above and fill in your method's details  
3. **✅ Validate** — Ensure all required fields are included
4. **🔄 Auto-Process** — The filename automatically identifies the method in our system

### 📏 **Naming Convention**

- ✅ Use lowercase letters, numbers, hyphens, and underscores
- ✅ Base filename on the method name for easy identification
- ✅ Keep it descriptive but concise

**Examples:**
- `Method: cPCA` → `cpca.yaml`
- `Method: GEARS` → `gears.yaml` 
- `Method: scGEN` → `scgen.yaml`
- `Method: AttentionPert` → `attentionpert.yaml`

---

## 📊 **Field Requirements**

### 🔴 **Required Fields**

| Field | Description | Example |
|-------|-------------|---------|
| `Method` | 🏷️ Official method name | `GEARS` |
| `Year` | 📅 Publication year | `2023` |
| `Description` | 📄 Detailed method description | `Multi-line text explaining the approach...` |
| `Publication` | 📖 DOI or publication URL | `https://doi.org/10.1038/...` |
| `Code Availability` | 💻 Code repository link or `'-'` | `https://github.com/user/repo` |
| `Published` | ✅ Publication status | `true` or `false` |
| `Task` | 🎯 List of computational tasks | `[Perturbation Prediction, ...]` |

### 🟡 **Optional Fields**

| Field | Description | Example |
|-------|-------------|---------|
| `Model` | 🔧 Underlying frameworks | `[VAE, GNN]` |
| `Inspired by` | 📚 Related references | `[PCA, NMF]` |

---

## 🔧 **Technical Details**

### 🔄 **Duplicate Handling**

If two methods share the same name, our system automatically handles this:
- **First occurrence**: `method.yaml`
- **Second occurrence**: `method_1.yaml`  
- **Third occurrence**: `method_2.yaml`

### ✅ **Validation**

- All YAML files are automatically validated for syntax
- Required fields are checked during the build process
- Unknown task categories trigger warnings (see `tasklist.txt`)

### 🚀 **Integration** 

- Files are automatically discovered by `generate_methods.py`
- Changes trigger automatic documentation rebuilds
- No manual intervention needed after file creation

---

<div align="center">

**🎉 Ready to contribute? Check our [📋 Contribution Guidelines](../docs/source/contribute.rst)!**

*Thank you for helping grow the single-cell methods catalog! 🙏*

</div>

