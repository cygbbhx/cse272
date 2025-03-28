#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;

    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Real h_dot_in = dot(half_vector, dir_in);

    // Q4.4
    // Real F0 = pow((1 - eta) / (1 + eta), 2);
    // Real n_dot_in = dot(frame.n, dir_in);
    // Real cos_theta_t_2 = 1 - (1 - pow(n_dot_in, 2)) / pow(eta, 2);

    // Real F = Real(1);

    // if (cos_theta_t_2 > 0) {
    //     F = F0 + (1 - F0) * pow(1 - sqrt(cos_theta_t_2), 5);
    // }

    Real F = fresnel_dielectric(h_dot_in, eta);

    Spectrum baseColor = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 h_l = to_local(frame, half_vector);
    Vector3 w_l_in = to_local(frame, dir_in);
    Vector3 w_l_out = to_local(frame, dir_out);

    Real D_g = D_metal(h_l, roughness, anisotropic);
    Real G_g = G_metal(w_l_in, roughness, anisotropic) * G_metal(w_l_out, roughness, anisotropic);

    if (reflect) {
        return baseColor * F *  D_g * G_g / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        return sqrt(baseColor) * (1 - F) * D_g * G_g * fabs(dot(half_vector, dir_out) * dot(half_vector, dir_in))
            / (fabs(dot(frame.n, dir_in)) * pow(dot(half_vector, dir_in) + eta * dot(half_vector, dir_out), 2));
    }
}

Real pdf_sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    assert(eta > 0);

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.

    // TODO: MODIFY F/D/G VALUE FOR GLASS
    Real h_dot_in = dot(half_vector, dir_in);

    // Q4.4
    // Real F0 = pow((1 - eta) / (1 + eta), 2);
    // Real n_dot_in = dot(frame.n, dir_in);
    // Real cos_theta_t_2 = 1 - (1 - pow(n_dot_in, 2)) / pow(eta, 2);

    // Real F = Real(1);

    // if (cos_theta_t_2 > 0) {
    //     F = F0 + (1 - F0) * pow(1 - sqrt(cos_theta_t_2), 5);
    // }

    Real F = fresnel_dielectric(h_dot_in, eta);

    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 h_l = to_local(frame, half_vector);
    Vector3 w_l_in = to_local(frame, dir_in);

    Real D = D_metal(h_l, roughness, anisotropic);
    Real G_in = G_metal(w_l_in, roughness, anisotropic);

    if (reflect) {
        return (F * D * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    // TODO: ANISOTROPIC VALUE
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    // Sample a micro normal and transform it to world space -- this is our half-vector.
    // TODO: SAMPLE VISIBLE NORMALS -> TAKES 2 ALPHA VALUES (A_X, A_Y)
    auto [alpha_x, alpha_y] = compute_alpha(roughness, anisotropic);
    
    Vector3 local_dir_in = to_local(frame, dir_in);
    Vector3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha_x, alpha_y, rnd_param_uv);

    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }
    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    Real h_dot_in = dot(half_vector, dir_in);

    // Q4.4
    // Real F0 = pow((1 - eta) / (1 + eta), 2);
    // Real n_dot_in = dot(frame.n, dir_in);
    // Real cos_theta_t_2 = 1 - (1 - pow(n_dot_in, 2)) / pow(eta, 2);

    // Real F = Real(1);

    // if (cos_theta_t_2 > 0) {
    //     F = F0 + (1 - F0) * pow(1 - sqrt(cos_theta_t_2), 5);
    // }

    Real F = fresnel_dielectric(h_dot_in, eta);


    if (rnd_param_w <= F) {
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        return BSDFSampleRecord{reflected, Real(0) /* eta */, roughness};
        // Reflection
    } else {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0) {
            // Total internal reflection
            // This shouldn't really happen, as F will be 1 in this case.
            return {};
        }
        // flip half_vector if needed
        if (h_dot_in < 0) {
            half_vector = -half_vector;
        }
        Real h_dot_out= sqrt(h_dot_out_sq);
        Vector3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
        return BSDFSampleRecord{refracted, eta, roughness};
    }
}

TextureSpectrum get_texture_op::operator()(const DisneyGlass &bsdf) const {
    return bsdf.base_color;
}
