# .Rprofile for Proteoforms Project
# This file is automatically loaded when R starts in this directory

# Activate renv for this project
if (file.exists("renv/activate.R")) {
  source("renv/activate.R")
  cat("✅ renv activated for Proteoforms project\n")
} else {
  cat("⚠️ renv not initialized. Run source('setup_env.R') to set up.\n")
}

# Set project-specific options
options(
  repos = c(CRAN = "https://cloud.r-project.org/"),
  max.print = 100,
  digits = 4,
  scipen = 999,
  BioC_mirror = "https://bioconductor.org"
)

# Welcome message
cat("🧬 Proteoforms Project R Environment\n")
cat("📁 Working directory:", getwd(), "\n")
cat("📦 R version:", R.version.string, "\n")

# Helper functions
show_packages <- function() {
  cat("📋 Installed packages:\n")
  installed <- installed.packages()[, c("Package", "Version")]
  print(head(installed, 20))
  if (nrow(installed) > 20) {
    cat("... and", nrow(installed) - 20, "more packages\n")
  }
}

# Auto-load commonly used packages for interactive sessions
if (interactive()) {
  suppressMessages({
    if (require("data.table", quietly = TRUE)) {
      cat("📊 data.table loaded\n")
    }
  })
}

cat("Type 'show_packages()' to see installed packages\n")
cat("=" , rep("=", 40), "\n")
