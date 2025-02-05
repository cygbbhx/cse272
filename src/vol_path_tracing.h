#pragma once

// The simplest volumetric renderer: 
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        // Hit background. Account for the environment map if needed.
        if (has_envmap(scene)) {
            const Light &envmap = get_envmap(scene);
            return emission(envmap,
                            -ray.dir, // pointing outwards from light
                            ray_diff.spread,
                            PointAndNormal{}, // dummy parameter for envmap
                            scene);
        }
        return make_zero_spectrum();
    }

    PathVertex vertex = *vertex_;
    // TODO: fix to  camera exterior ID 
    int camera_id = scene.camera.medium_id;
    Medium cur_medium = scene.media[camera_id];
    Spectrum sigma_a = get_sigma_a(cur_medium, vertex_->position);
    Spectrum current_path_throughput = fromRGB(Vector3{1, 1, 1});

    // We hit a light immediately. 
    if (is_light(scene.shapes[vertex.shape_id])) {
        Real t = distance(vertex.position, ray.org);
        Spectrum transmittance = exp(-sigma_a * t);
        // consider vacuum case (where medium id is -1)
        if (camera_id == -1) {
            return make_zero_spectrum();
        }
        return current_path_throughput *
            emission(vertex, -ray.dir, scene) * transmittance;
    }

    return make_zero_spectrum();
}

// The second simplest volumetric renderer: 
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = init_ray_differential(w, h);

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    Real t_max;
    if (!vertex_) {
        t_max = infinity<Real>();
    }
    else {
        PathVertex vertex = *vertex_;
        t_max = distance(vertex.position, ray.org);
    }

    int camera_id = scene.camera.medium_id;
    int cur_medium_id = camera_id;
    Medium cur_medium = scene.media[camera_id];

    Real sigma_a = avg(get_sigma_a(cur_medium, ray.org));
    Real sigma_s = avg(get_sigma_s(cur_medium, ray.org));
    Real sigma_t = sigma_a + sigma_s;

    Real u = next_pcg32_real<Real>(rng);
    Real t = -log(1 - u) / sigma_t; 

    if (t < t_max){
        Real trans_pdf = exp(-sigma_t * t) * sigma_t;
        Real transmittance = exp(-sigma_t * t);
        PhaseFunction rho = get_phase_function(cur_medium);

        Spectrum current_path_throughput = fromRGB(Vector3{1, 1, 1});
        Spectrum Ls1 = make_zero_spectrum();
        Real exp_val;
        Vector3 p = ray.org + t * ray.dir;

        // Equation 7
        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, p, light_uv, shape_w, scene);
        Vector3 omega_prime = normalize(point_on_light.position - p);
        Real w1 = 0;

        {
            Real Jacobian = 0;
            exp_val = exp(-sigma_t * distance(point_on_light.position, p));

            Ray shadow_ray{p, omega_prime, 
                            get_shadow_epsilon(scene),
                            (1 - get_shadow_epsilon(scene)) *
                                distance(point_on_light.position, p)};
            
            if (!occluded(scene, shadow_ray)) {
                Jacobian = fabs(dot(omega_prime, point_on_light.normal)) /
                                        distance_squared(point_on_light.position, p);
            }

            Real p1 = light_pmf(scene, light_id) *
                pdf_point_on_light(light, point_on_light, p, scene);

            if (Jacobian > 0 && p1 > 0) {
                Vector3 dir_view = -ray.dir;
                Spectrum rho_val = eval(rho, dir_view, omega_prime);
                Spectrum Le = emission(light, -omega_prime, Real(0), point_on_light, scene);

                Ls1 = Jacobian * rho_val * Le * exp_val;
                Ls1 /= p1;
            }
        }
        return (transmittance / trans_pdf) * sigma_s * Ls1;

    }
    else {
        Real trans_pdf = exp(-sigma_t * t_max);
        Real transmittance = exp(-sigma_t * t_max);
        Spectrum Le = make_zero_spectrum();

        // We hit a light immediately. 
        if (vertex_){
            PathVertex vertex = *vertex_;
            if (is_light(scene.shapes[vertex.shape_id])) {
                // consider vacuum case (where medium id is -1)
                if (camera_id == -1) {
                    return make_zero_spectrum();
                }
                Le = emission(vertex, -ray.dir, scene);
            }

        }
        return (transmittance / trans_pdf) * Le;
    }
}


// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The fifth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}

// The final volumetric renderer: 
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}
