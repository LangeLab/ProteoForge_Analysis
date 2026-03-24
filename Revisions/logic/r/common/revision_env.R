get_revision_context <- function(script_path) {
    repo_root <- normalizePath(file.path(dirname(script_path), '..', '..', '..', '..', '..'))
    renv_activate <- file.path(repo_root, 'renv', 'activate.R')
    if (file.exists(renv_activate)) {
        source(renv_activate)
    }

    list(
        repo_root = repo_root,
        revisions_root = file.path(repo_root, 'Revisions'),
        revision_output_root = file.path(repo_root, 'Revisions', 'outputs', 'simulation'),
        simulation_data_root = file.path(repo_root, 'Simulation', 'data')
    )
}