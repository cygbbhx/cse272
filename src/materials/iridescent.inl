//"A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence
//Laurent Belcour, Pascal Barla
//ACM Transactions on Graphics (proc. of SIGGRAPH 2017)
//
//May 2017"

#include "../microfacet.h"

Spectrum eval_op::operator()(const Iridescent &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);    
    Real alpha = roughness * roughness;

    Real d = bsdf.d;
    Real eta2 = bsdf.eta2;
    Real eta3 = bsdf.eta3;
    Real kappa3 = bsdf.kappa3;

    Real n_dot_l = dot(frame.n, dir_in);
    Real n_dot_v = dot(frame.n, dir_out);

    if (n_dot_l < 0 || n_dot_v < 0) {
        return make_const_spectrum(0.0);
    }

    Vector3 half_vector;
    half_vector = normalize(dir_in + dir_out);

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
    Real OPD = 2 * eta2 * d * cos_theta_2; // Or D_inc * cos_theta_2 (D_inc = 2 * eta2 * d)
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

    I = saturate(XYZ_to_RGB(I));

    Real D = GTR2(dot(frame.n, half_vector), roughness);
    Real G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha);
    Real G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha);
    Real G = G_in * G_out;

    Spectrum f = (D * G * I) / (4.0 * fabs(n_dot_l));
    return f;
}

Real pdf_sample_bsdf_op::operator()(const Iridescent &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
    dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    Vector3 half_vector = normalize(dir_in + dir_out);
    Real n_dot_out = dot(frame.n, dir_out);
    Real n_dot_h = dot(frame.n, half_vector);
    Real h_dot_out = dot(half_vector, dir_out);
    if (n_dot_out <= 0 || n_dot_h <= 0 || h_dot_out <= 0) {
        return 0;
    }

    Real roughness = eval(
        bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha = roughness * roughness;

    Real D = GTR2(n_dot_h, roughness);
    Real G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha);

    return (D * G_in) / (4 * dot(dir_in, frame.n));
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const Iridescent &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    
    Vector3 local_dir_in = to_local(vertex.shading_frame, dir_in);
    constexpr Real min_alpha = Real(0.0001);
    Real alpha = max(roughness * roughness, min_alpha);

    Vector3 local_micro_normal =
        sample_visible_normals(local_dir_in, alpha, rnd_param_uv);

    // Transform the micro normal to world space
    Vector3 half_vector = to_world(vertex.shading_frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected, Real(0) /* eta */, roughness /* roughness */};
}

TextureSpectrum get_texture_op::operator()(const Iridescent &bsdf) const {
    return bsdf.base_color;
}
