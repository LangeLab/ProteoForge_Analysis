# Simplified R Environment Setup for the project
## Uses renv for environment management and pak for package installation

# Core R requirements
required_r_version <- "4.5.0"

# Check R version
if (getRversion() < required_r_version) {
  stop(paste("R version", required_r_version, "or higher is required. Current version:", getRversion()))
}

# Install renv if not already installed
if (!require("renv", quietly = TRUE)) {
  install.packages("renv")
}

# Install pak if not already installed
if (!require("pak", quietly = TRUE)) {
  install.packages("pak")
}

# Check if renv is already initialized
if (file.exists("renv.lock")) {
  cat("renv.lock exists. Activating the project...\n")
  renv::activate()
} else if (file.exists("renv") && file.exists(".Rprofile")) {
  cat("renv is activated but renv.lock is missing. Re-creating renv.lock...\n")
  renv::activate()
  renv::snapshot()
} else {
  cat("Initializing renv for this project...\n")
  renv::init()
  renv::settings$use.cache(FALSE)
  renv::settings$library("renv/library")
}

# Ensure base packages are loaded
if (!requireNamespace("base", quietly = TRUE)) {
    stop("Base R packages are not loaded. Please check your R installation.")
}

# Centralized list of all required packages
# pak can handle different sources using special syntax.
project_dependencies <- c(
  # CRAN Packages
  "data.table",
  "arrow",
  "languageserver",

  # Bioconductor Packages
  "bioc::qvalue",

  # GitHub Packages
  "github::jessegmeyerlab/PeCorA",
  "github::CCprofiler/CCprofiler@proteoformLocationMapping"
)

cat("Installing/updating all project dependencies with pak...\n")
# This single command handles everything: resolving, downloading, and installing.
# It's fast, caches aggressively, and provides clear error messages.
pak::pkg_install(project_dependencies)

# Snapshot the environment
cat("Snapshotting the environment...\n")
renv::snapshot()

cat("✅ R environment setup complete!\n")