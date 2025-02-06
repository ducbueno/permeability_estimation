#!/usr/bin/awk -f

{
    gsub(/TAU/, tau)
    gsub(/VL/, vl)
    gsub(/TSMAX/, tsmax)
    gsub(/ANALYSIS_INTERVAL/, tsmax)
    gsub(/TOL/, tol)
    gsub(/FORCE/, force)
    gsub(/SAMPLE_SIZE/, sample_size)
    gsub(/SUBDOMAIN_SIZE/, subdomain_size)

    if (mirror == "True" && (/InletLayers/ || /OutletLayers/)) {
        next
    }

    if (vis == "False" && (/write_silo/ || /save_velocity/ || /save_pressure/)) {
        next
    }

    print
}
