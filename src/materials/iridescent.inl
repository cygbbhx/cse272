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

    const Real r =  2.3646381*I[0] - 0.8965361*I[1] - 0.4680737*I[2];
    const Real g = -0.5151664*I[0] + 1.4264000*I[1] + 0.0887608*I[2];
    const Real b =  0.0052037*I[0] - 0.0144081*I[1] + 1.0092106*I[2];
    I[0] = r;
    I[1] = g;
    I[2] = b;
    I = saturate(I);
    // I = saturate(XYZ_to_RGB(I));

    Real D = GTR2(dot(frame.n, half_vector), roughness);
    Real G = smithG_GGX(n_dot_l, n_dot_v, alpha);

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

    return (D * n_dot_h) / (4 * h_dot_out);
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
    Real alpha = roughness * roughness;
    
    // Appendix B.2 Burley's note
    Real alpha2 = alpha * alpha;

    Real phi = 2 * c_PI * rnd_param_uv[0];
    Real cos_theta = sqrt((1.0 - rnd_param_uv[1]) / (1.0 + (alpha2 - 1.0) * rnd_param_uv[1]));
    Real sin_theta = sqrt(max(1e-5, 1.0 - cos_theta * cos_theta));

    Vector3 local_micro_normal{
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    };
    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);

    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected, Real(0) /* eta */, roughness
    };
}

TextureSpectrum get_texture_op::operator()(const Iridescent &bsdf) const {
    return bsdf.base_color;
}
