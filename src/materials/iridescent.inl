//"A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence
//Laurent Belcour, Pascal Barla
//ACM Transactions on Graphics (proc. of SIGGRAPH 2017)
//
//May 2017"

#include "../microfacet.h"

Spectrum fresnel_iridescent(
    const Real &cos_theta_1,
    const Real &eta,
    const Real &d,
    const Real &eta2,
    const Real &eta3,
    const Real &kappa3,
    const bool &isConductorBase
) {
    // First interface
    Vector2 R12; Vector2 phi12;
    std::tie(R12, phi12) = F_dielectric(cos_theta_1, 1.0 / eta); 
    
    Vector2 R21 = R12;
    Vector2 T121 = 1.0 - R12;
    Vector2 phi21 = c_PI - phi12;

    // Second interface
    Real cos_theta_2_sqr = 1.0 - sqr(1.0 / eta) * (1.0 - sqr(cos_theta_1));
    Vector2 R23; Vector2 phi23;
    
    if (cos_theta_2_sqr < 0) {
        return make_const_spectrum(1.0);
    }
    Real cos_theta_2 = sqrt(cos_theta_2_sqr);

    if (isConductorBase) {
        std::tie(R23, phi23) = F_conductor(cos_theta_2, eta2, eta3, kappa3);
    } else {
        std::tie(R23, phi23) = F_dielectric(cos_theta_2, eta);
    }

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

    return saturate(I);
}


Spectrum eval_op::operator()(const Iridescent &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    bool isConductorBase = bsdf.isConductorBase;

    if (isConductorBase && (dot(vertex.geometric_normal, dir_in) < 0 ||
                            dot(vertex.geometric_normal, dir_out) < 0)) {
        // No light below the surface
        return make_zero_spectrum();
    }

    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);    
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real alpha = roughness * roughness;

    Real n_dot_l = dot(frame.n, dir_in);
    Real n_dot_v = dot(frame.n, dir_out);

    if (isConductorBase && (n_dot_l < 0 || n_dot_v < 0)) {
        return make_zero_spectrum();
    }

    Vector3 half_vector = normalize(dir_in + dir_out);
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 || isConductorBase ? bsdf.eta2 : 1 / bsdf.eta2;
    
    if (!isConductorBase) {
        if (!reflect) {
            half_vector = normalize(dir_in + dir_out * eta);
        }
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }
    }

    Real h_dot_in = dot(half_vector, dir_in);
    Spectrum I = fresnel_iridescent(h_dot_in, eta, bsdf.d, bsdf.eta2, bsdf.eta3, bsdf.kappa3, isConductorBase);

    Real D = GTR2(dot(frame.n, half_vector), roughness);
    Real G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha);
    Real G_out = smith_masking_gtr2(to_local(frame, dir_out), alpha);
    Real G = G_in * G_out;
    
    if (reflect || isConductorBase) {
        return I * D * G / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        Real eta_factor = dir == TransportDirection::TO_LIGHT ? (1 / (eta * eta)) : 1;
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        return (eta_factor * (1 - I) * D * G * eta * eta * fabs(h_dot_out * h_dot_in)) / 
                (fabs(dot(frame.n, dir_in)) * sqrt_denom * sqrt_denom);
    }
}

Real pdf_sample_bsdf_op::operator()(const Iridescent &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    bool isConductorBase = bsdf.isConductorBase;

    if (isConductorBase && (dot(vertex.geometric_normal, dir_in) < 0 ||
                            dot(vertex.geometric_normal, dir_out) < 0)) {
        // No light below the surface
        return 0;
    }

    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    Real eta = dot(vertex.geometric_normal, dir_in) > 0 || isConductorBase ? bsdf.eta2 : 1 / bsdf.eta2;

    Vector3 half_vector = normalize(dir_in + dir_out);

    if (isConductorBase) {
        Real n_dot_out = dot(frame.n, dir_out);
        Real n_dot_h = dot(frame.n, half_vector);
        Real h_dot_out = dot(half_vector, dir_out);
        if (n_dot_out <= 0 || n_dot_h <= 0 || h_dot_out <= 0) {
            return 0;
        }
    } else {        
        if (!reflect) {
            half_vector = normalize(dir_in + dir_out * eta);
        }
        if (dot(half_vector, frame.n) < 0) {
            half_vector = -half_vector;
        }
    }
    Real n_dot_h = dot(frame.n, half_vector);

    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);    
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real alpha = roughness * roughness;

    Real h_dot_in = dot(half_vector, dir_in);
    
    Real D = GTR2(n_dot_h, roughness);
    Real G_in = smith_masking_gtr2(to_local(frame, dir_in), alpha);
    // Spectrum I = fresnel_iridescent(h_dot_in, eta, bsdf);
    // Real F = avg(I);
    Real F = isConductorBase ? Real(1.0) : fresnel_dielectric(h_dot_in, eta);

    if (reflect || isConductorBase) {
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
    bool isConductorBase = bsdf.isConductorBase;

    if (isConductorBase && (dot(vertex.geometric_normal, dir_in) < 0)) {
        // No light below the surface
        return {};
    }

    Real eta = dot(vertex.geometric_normal, dir_in) > 0 || isConductorBase ? bsdf.eta2 : 1 / bsdf.eta2;

    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    constexpr Real min_alpha = Real(0.0001);
    Real alpha = max(roughness * roughness, min_alpha);

    Vector3 local_dir_in = to_local(frame, dir_in);
    Vector3 local_micro_normal = sample_visible_normals(local_dir_in, alpha, rnd_param_uv);

    Vector3 half_vector = to_world(frame, local_micro_normal);
    if (!isConductorBase && (dot(half_vector, frame.n) < 0)) {
        half_vector = -half_vector;
    }
    Real h_dot_in = dot(half_vector, dir_in);

    // Spectrum I = fresnel_iridescent(h_dot_in, eta, bsdf);
    // Real F = avg(I);
    Real F = fresnel_dielectric(h_dot_in, eta);
    
    if (rnd_param_w <= F || isConductorBase) {
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        return BSDFSampleRecord{reflected, Real(0) /* eta */, roughness};
        // Reflection
    } else {

        Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0) {
            return {};
        }
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
