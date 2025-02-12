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
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

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
    Spectrum sigma_a = get_sigma_a(cur_medium, ray.org);
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
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

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

int update_medium_id(Ray ray, PathVertex isect, int &old_medium_id) {
    int medium_id = old_medium_id;
    if (isect.interior_medium_id != isect.exterior_medium_id) {
        if (dot(ray.dir, isect.geometric_normal) > 0) {
            medium_id = isect.exterior_medium_id;
        }
        else {
            medium_id = isect.interior_medium_id;
        }
    }
    return medium_id;
}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};


    Spectrum current_path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    
    int camera_id = scene.camera.medium_id;
    Medium cur_medium = scene.media[camera_id];
    int cur_medium_id = camera_id;

    int max_depth = scene.options.max_depth;

    while (true) {
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        PathVertex vertex;
        bool scatter = false;

        Real transmittance = 1;
        Real trans_pdf = 1;

        Real t_max;
        if (!vertex_) {
            t_max = infinity<Real>();
        }
        else {
            vertex = *vertex_;
            t_max = distance(vertex.position, ray.org);
        }

        if (cur_medium_id >= 0) {
            // sample t s.t. p(t) ~ exp(-sigma_t * t)
            cur_medium = scene.media[cur_medium_id];

            Real sigma_a = avg(get_sigma_a(cur_medium, ray.org));
            Real sigma_s = avg(get_sigma_s(cur_medium, ray.org));
            Real sigma_t = sigma_a + sigma_s;

            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(1 - u) / sigma_t; 

            // compute transmittance and trans_pdf
            // if t < t_hit, set scatter = True
            if (t < t_max) {
                transmittance = exp(-sigma_t * t);
                trans_pdf = exp(-sigma_t * t) * sigma_t;
                scatter = true;
                ray.org = ray.org + t * ray.dir;
            } else {
                trans_pdf = exp(-sigma_t * t_max);
                transmittance = exp(-sigma_t * t_max);
                ray.org = ray.org + t_max * ray.dir;
            }
        } 

        current_path_throughput *= (transmittance / trans_pdf);
        if (not scatter) {
            // reach a surface, include emission
            Spectrum Le = make_zero_spectrum();
            if (vertex_){
                if (is_light(scene.shapes[vertex.shape_id])) {
                    Le = emission(vertex, -ray.dir, scene);
                }

            }
            radiance += current_path_throughput * Le;
        }
        if (bounces == max_depth - 1 and max_depth != -1) {
            // reach maximum bounces
            break;
        }
        if (not scatter and vertex_) {
            if (vertex.material_id == -1){
                // index-matching interface, skip through it
                ray = Ray{ray.org, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
                cur_medium_id = update_medium_id(ray, vertex, cur_medium_id);
                bounces += 1;
                continue;
            }
        }
        // sample next direct & update path current_path_throughput
        if (scatter) {
            PhaseFunction rho = get_phase_function(cur_medium);
            Spectrum sigma_s = get_sigma_s(cur_medium, ray.org);
            Vector2 phase_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

            std::optional <Vector3> next_dir_ = sample_phase_function(rho, -ray.dir, phase_uv);

            if (!next_dir_){
                break;
            }
            const Vector3 next_dir = *next_dir_;
            Spectrum phase_val = eval(rho, -ray.dir, next_dir);
            Real phase_pdf = pdf_sample_phase(rho, -ray.dir, next_dir);
            current_path_throughput *= (phase_val / phase_pdf) * sigma_s;
            // update ray.dir
            // ...
            Ray next_ray{ray.org, next_dir, get_intersection_epsilon(scene), infinity<Real>()};
            ray = next_ray;
        } else {
            // Hit a surface
            break;
        }
        // Russian roulette
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth){
            rr_prob = min(max(current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            else {
                current_path_throughput /= rr_prob;
            }
        }
        bounces += 1;
    }
    return radiance;
}

Spectrum next_event_estimation(const Scene &scene, const int &cur_medium_id, pcg32_state &rng, int &bounces, const Ray &ray, bool &scatter, const PathVertex &vertex) {
    Vector3 p = ray.org;
    RayDifferential ray_diff{Real(0), Real(0)};
    
    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
    Real light_w = next_pcg32_real<Real>(rng);
    Real shape_w = next_pcg32_real<Real>(rng);
    int light_id = sample_light(scene, light_w);
    const Light &light = scene.lights[light_id];
    PointAndNormal p_prime = sample_point_on_light(light, p, light_uv, shape_w, scene);
    
    Spectrum T_light = make_const_spectrum(1); 
    Real p_trans_dir = 1;
    int shadow_medium_id = cur_medium_id, shadow_bounces = 0;
    Vector3 p_tmp = p;
    
    while (true) {
        Vector3 dir_light = normalize(p_prime.position - p_tmp);
        Ray shadow_ray{p_tmp, dir_light, get_shadow_epsilon(scene), (1 - get_shadow_epsilon(scene)) * distance(p_prime.position, p_tmp)};
        
        std::optional<PathVertex> shadow_vertex_ = intersect(scene, shadow_ray, ray_diff);
        PathVertex shadow_vertex;
        if (shadow_vertex_) shadow_vertex = *shadow_vertex_;
        Real next_t = distance(p_tmp, shadow_vertex_ ? shadow_vertex.position : p_prime.position);
        
        if (shadow_medium_id >= 0) {
            const Medium &cur_medium = scene.media[shadow_medium_id];
            Spectrum sigma_t = get_sigma_a(cur_medium, p) + get_sigma_s(cur_medium, p);
            Real transmittance = exp(-sigma_t[0] * next_t);

            T_light *= transmittance;
            p_trans_dir *= transmittance;
        }
        
        if (!shadow_vertex_) break;
        if (shadow_vertex.material_id >= 0) return make_zero_spectrum();
        
        shadow_medium_id = update_medium_id(shadow_ray, *shadow_vertex_, shadow_medium_id);
        p_tmp += next_t * dir_light;

        if (++shadow_bounces + bounces + 1 >= scene.options.max_depth && scene.options.max_depth != -1) return make_zero_spectrum();
    }

    if (max(T_light) <= 0) return make_zero_spectrum();
    
    Spectrum contrib = make_zero_spectrum();
    Vector3 dir_light = normalize(p_prime.position - p);
    Real G = max(-dot(dir_light, p_prime.normal), Real(0)) / distance_squared(p_prime.position, p);
    Real pdf_nee = light_pmf(scene, light_id) * pdf_point_on_light(light, p_prime, p, scene);
    if (G <= 0 || pdf_nee <= 0) return make_zero_spectrum();
    
    Spectrum f, sigma_s = make_const_spectrum(1);
    Real p2 = 0;

    if (scatter) {
        const Medium &cur_medium = scene.media[cur_medium_id];
        PhaseFunction rho = get_phase_function(cur_medium);
        f = eval(rho, -ray.dir, dir_light);
        sigma_s = get_sigma_s(cur_medium, p);
        p2 = pdf_sample_phase(rho, -ray.dir, dir_light) * G;
    } else {
        f = eval(scene.materials[vertex.material_id], -ray.dir, dir_light, vertex, scene.texture_pool);
    }
    
    p2 *= p_trans_dir;
    Spectrum Le = emission(light, -dir_light, Real(0), p_prime, scene);
    Real w = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + p2 * p2);
    Spectrum C = T_light * G * f * Le / pdf_nee;
    
    return w * C * sigma_s;
}

// The fourth volumetric renderer: 
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    // Homework 2: implememt this!
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);    
    RayDifferential ray_diff{Real(0), Real(0)};

    Spectrum current_path_throughput = make_const_spectrum(1), radiance = make_zero_spectrum();
    int bounces = 0, cur_medium_id = scene.camera.medium_id, max_depth = scene.options.max_depth;

    Real dir_pdf = 0;
    Vector3 nee_p_cache;
    Real multi_trans_pdf = 1;
    bool never_scatter = true;
     while (true) {
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        PathVertex vertex;
        if (vertex_) vertex = *vertex_;

        bool scatter = false;
        Real transmittance = 1, trans_pdf = 1, t_max = vertex_ ? distance(vertex.position, ray.org) : infinity<Real>();
        
        if (cur_medium_id >= 0) {
            const Medium &cur_medium = scene.media[cur_medium_id];
            Real sigma_t = avg(get_sigma_a(cur_medium, ray.org)) + avg(get_sigma_s(cur_medium, ray.org));
            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(u) / sigma_t;
            
            if (t < t_max) {
                transmittance = exp(-sigma_t * t);
                trans_pdf = sigma_t * transmittance;
                scatter = true;
                never_scatter = false;
                ray.org += t * ray.dir;
            } else {
                transmittance = exp(-sigma_t * t_max);
                trans_pdf = exp(-sigma_t * t_max);
                ray.org += t_max * ray.dir;
            }
        }
        
        multi_trans_pdf *= trans_pdf;
        current_path_throughput *= transmittance / trans_pdf;
        
        if (!scatter && !vertex_) break;
        
        if (!scatter) {
            if (vertex_ && is_light(scene.shapes[vertex.shape_id])) {
                Spectrum Le = emission(*vertex_, -ray.dir, scene);
                if (never_scatter) {
                    radiance += current_path_throughput * Le;
                } else {
                    int light_id = get_area_light_id(scene.shapes[vertex.shape_id]);
                    const Light &light = scene.lights[light_id];
                    PointAndNormal light_point{vertex.position, vertex.geometric_normal};
                    
                    Real pdf_nee = pdf_point_on_light(light, light_point, nee_p_cache, scene);
                    Real p_nee = light_pmf(scene, light_id) * pdf_nee;
                        
                    Vector3 light_dir = normalize(vertex.position - nee_p_cache);
                    Real G = fabs(dot(vertex.geometric_normal, light_dir)) /
                        distance_squared(nee_p_cache, vertex.position);

                    Real dir_pdf_ = dir_pdf * multi_trans_pdf * G;
                    Real w = (dir_pdf_ * dir_pdf_) / (dir_pdf_ * dir_pdf_ + p_nee * p_nee);
                    radiance += w * current_path_throughput * Le;
                }
            }
        }
        if (bounces == max_depth - 1 and max_depth != -1) {
            // reach maximum bounces
            break;
        }
        if (!scatter && vertex_) {
            if (vertex.material_id == -1){
                // index-matching interface, skip through it
                ray = Ray{vertex.position, ray.dir, get_intersection_epsilon(scene), infinity<Real>()};
                cur_medium_id = update_medium_id(ray, vertex, cur_medium_id);
                bounces += 1;
                continue;
            }
        }

        // cache the position for MIS
        nee_p_cache = ray.org;
        multi_trans_pdf = 1;
        radiance += current_path_throughput * next_event_estimation(scene, cur_medium_id, rng, bounces, ray, scatter, vertex);
        
        if (scatter) {
            const Medium &cur_medium = scene.media[cur_medium_id];
            PhaseFunction rho = get_phase_function(cur_medium);
            Spectrum sigma_s = get_sigma_s(cur_medium, ray.org);
            Vector2 phase_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};

            std::optional <Vector3> next_dir_ = sample_phase_function(rho, -ray.dir, phase_uv);

            if (!next_dir_) break;

            const Vector3 next_dir = *next_dir_;
            Spectrum phase_val = eval(rho, -ray.dir, next_dir);
            Real phase_pdf = pdf_sample_phase(rho, -ray.dir, next_dir);

            current_path_throughput *= (phase_val / phase_pdf) * sigma_s;
            
            dir_pdf = phase_pdf;
            Ray next_ray{ray.org, next_dir, get_intersection_epsilon(scene), infinity<Real>()};
            ray = next_ray;
        } else {
            // Hit a surface
            break;
        }
        // Russian roulette
        Real rr_prob = 1;
        if (bounces >= scene.options.rr_depth){
            rr_prob = min(max(current_path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            else {
                current_path_throughput /= rr_prob;
            }
        }
        bounces += 1;
    }
    return radiance;
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
