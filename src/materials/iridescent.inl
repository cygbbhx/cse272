//"A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence
//Laurent Belcour, Pascal Barla
//ACM Transactions on Graphics (proc. of SIGGRAPH 2017)
//
//May 2017"

#include "../microfacet.h"

Spectrum eval_op::operator()(const Iridescent &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);    

    Real d = bsdf.d;
    Real eta2 = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta2 : 1 / bsdf.eta2;
    Real eta3 = bsdf.eta3;
    Real kappa3 = bsdf.kappa3;

    Real n_dot_l = dot(frame.n, dir_in);
    Real n_dot_v = dot(frame.n, dir_out);

    if (n_dot_l < 0 || n_dot_v < 0) {
        return make_const_spectrum(0.0);
    }

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta2);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // TODO: Force eta2 -> 1 when d -> 0
    
    Real cos_theta_1 = dot(half_vector, dir_in);
    Real cos_theta_2 = sqrt(1.0 - sqr(1.0 / eta2) * (1.0 - sqr(cos_theta_1)));

    // First interface
    Vector2 R12; Vector2 phi12;
    std::tie(R12, phi12) = F_dielectric(cos_theta_1, 1.0, eta2);    
    Vector2 R21 = R12;
    Vector2 T121 = 1.0 - R12;
    Vector2 phi21 = c_PI - phi12;

    // Second interface
    Vector2 R23; Vector2 phi23;
    std::tie(R23, phi23) = F_conductor(cos_theta_2, eta2, eta3, kappa3);

    // phase shift
    Real OPD = d * cos_theta_2;
    Vector2 phi2 = phi21 + phi23;

    // Compound terms
    Spectrum I = make_const_spectrum(0.0);
    Vector2 R123 = R12 * R23;
    Vector2 r123 = sqrt(R123);
    Vector2 Rs = sqr(T121) * R23 / (1 - R123);

    // Reflectance term for m=0
    Vector2 C0 = R12 + Rs;
    Spectrum S0 = eval_sensitivity(0.0, 0.0);
    I += depol(C0) * S0;

    // Reflectance term for m>0
    Vector2 Cm = Rs - T121;
    for (int m = 1; m < 4; ++m) {
        Cm *= r123;
        Spectrum SmS = 2.0 * eval_sensitivity(m * OPD, m * phi2.x);
        Spectrum SmP = 2.0 * eval_sensitivity(m * OPD, m * phi2.y);
        I += depolColor(Cm.x * SmS, Cm.y * SmP);
    }

    I = XYZ_to_RGB(I);

    Real D = GGX(cos_theta_1, roughness);
    Real G = smithG_GGX(n_dot_l, n_dot_v, roughness);
    Spectrum f = (D * G * I) / (4.0 * n_dot_l * n_dot_v);
    return f;
}

Real pdf_sample_bsdf_op::operator()(const Iridescent &bsdf) const {
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
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta2 : 1 / bsdf.eta2;
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

    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);

    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 h_l = to_local(frame, half_vector);
    Vector3 w_l_in = to_local(frame, dir_in);

    // Real D = D_metal(h_l, roughness, anisotropic);
    // Real G_in = G_metal(w_l_in, roughness, anisotropic);

    Real cos_theta_1 = dot(half_vector, dir_in);

    Real D = GGX(cos_theta_1, roughness);
    Real n_dot_l = dot(frame.n, dir_in);
    Real n_dot_v = dot(frame.n, dir_out);
    
    Real G_in = smithG1_GGX(n_dot_l, roughness);
    Real G = smithG_GGX(n_dot_l, n_dot_v, roughness);
    

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
        sample_bsdf_op::operator()(const Iridescent &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta2 : 1 / bsdf.eta2;
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
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

TextureSpectrum get_texture_op::operator()(const Iridescent &bsdf) const {
    return bsdf.base_color;
}
