#pragma once

#include "lajolla.h"
#include "spectrum.h"

/// A microfacet model assumes that the surface is composed of infinitely many little mirrors/glasses.
/// The orientation of the mirrors determines the amount of lights reflected.
/// The distribution of the orientation is determined empirically.
/// The distribution that fits the best to the data we have so far (which is not a lot of data)
/// is from Trowbridge and Reitz's 1975 paper "Average irregularity representation of a rough ray reflection",
/// wildly known as "GGX" (seems to stand for "Ground Glass X" https://twitter.com/CasualEffects/status/783018211130441728).
/// 
/// We will use a generalized version of GGX called Generalized Trowbridge and Reitz (GTR),
/// proposed by Brent Burley and folks at Disney (https://www.disneyanimation.com/publications/physically-based-shading-at-disney/)
/// as our normal distribution function. GTR2 is equivalent to GGX.

/// Schlick's Fresnel equation approximation
/// from "An Inexpensive BRDF Model for Physically-based Rendering", Schlick
/// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.2297&rep=rep1&type=pdf
/// See "Memo on Fresnel equations" from Sebastien Lagarde
/// for a really nice introduction.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
template <typename T>
inline T schlick_fresnel(const T &F0, Real cos_theta) {
    return F0 + (Real(1) - F0) *
        pow(max(1 - cos_theta, Real(0)), Real(5));
}

inline Real schlick_fresnel(Real F0, Vector3 n, Vector3 w) {
    return Real(1) + (F0 - Real(1)) * pow(Real(1) - fabs(dot(n, w)), Real(5));
}

/// Fresnel equation of a dielectric interface.
/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// n_dot_i: abs(cos(incident angle))
/// n_dot_t: abs(cos(transmission angle))
/// eta: eta_transmission / eta_incident
inline Real fresnel_dielectric(Real n_dot_i, Real n_dot_t, Real eta) {
    assert(n_dot_i >= 0 && n_dot_t >= 0 && eta > 0);
    Real rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
    Real rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
    Real F = (rs * rs + rp * rp) / 2;
    return F;
}

/// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
/// This is a specialized version for the code above, only using the incident angle.
/// The transmission angle is derived from 
/// n_dot_i: cos(incident angle) (can be negative)
/// eta: eta_transmission / eta_incident
inline Real fresnel_dielectric(Real n_dot_i, Real eta) {
    assert(eta > 0);
    Real n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
    if (n_dot_t_sq < 0) {
        // total internal reflection
        return 1;
    }
    Real n_dot_t = sqrt(n_dot_t_sq);
    return fresnel_dielectric(fabs(n_dot_i), n_dot_t, eta);
}

inline Real GTR2(Real n_dot_h, Real roughness) {
    Real alpha = roughness * roughness;
    Real a2 = alpha * alpha;
    Real t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
    return a2 / (c_PI * t*t);
}

inline Real GGX(Real n_dot_h, Real roughness) {
    return GTR2(n_dot_h, roughness);
}

inline std::pair<Real, Real> compute_alpha(Real roughness, Real anisotropic) {
    Real aspect = sqrt(1 - anisotropic * 0.9);
    Real alpha_min = 0.0001;
    Real alpha_x = fmax(alpha_min, (roughness * roughness) / aspect);
    Real alpha_y = fmax(alpha_min, (roughness * roughness) * aspect);
    
    return {alpha_x, alpha_y};
}

inline Real D_metal(Vector3 h_l, Real roughness, Real anisotropic) {
    auto [alpha_x, alpha_y] = compute_alpha(roughness, anisotropic);

    Real h_l_factor = ((h_l.x * h_l.x))/(alpha_x * alpha_x) + (h_l.y * h_l.y)/(alpha_y * alpha_y) + (h_l.z * h_l.z);
    Real D_m = 1 / (c_PI * alpha_x * alpha_y * (h_l_factor * h_l_factor));
    
    return D_m;
}

inline Real G_clearcoat(Vector3 w_l) {
    Real r = 0.25;
    Real Lamda = (sqrt(1 + ((w_l.x * r * w_l.x * r) + (w_l.y * r * w_l.y * r))/((w_l.z * w_l.z))) - 1) / 2;
    return 1 / (1 + Lamda);
}

inline Real G_metal(Vector3 w_l, Real roughness, Real anisotropic) {
    auto [alpha_x, alpha_y] = compute_alpha(roughness, anisotropic);

    Real lambda = (sqrt(1 + ((w_l.x*alpha_x * w_l.x*alpha_x) + (w_l.y*alpha_y*w_l.y*alpha_y)) / (w_l.z * w_l.z)) - 1) / 2;
    return 1 / (1 + lambda);
}

/// The masking term models the occlusion between the small mirrors of the microfacet models.
/// See Eric Heitz's paper "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
/// for a great explanation.
/// https://jcgt.org/published/0003/02/03/paper.pdf
/// The derivation is based on Smith's paper "Geometrical shadowing of a random rough surface".
/// Note that different microfacet distributions have different masking terms.
inline Real smith_masking_gtr2(const Vector3 &v_local, Real roughness) {
    Real alpha = roughness * roughness;
    Real a2 = alpha * alpha;
    Vector3 v2 = v_local * v_local;
    Real Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
    return 1 / (1 + Lambda);
}

inline Real smith_masking_gtr2(const Vector3 &v_local, Real alpha_x, Real alpha_y) {
    Real lambda = (sqrt(1 + ((v_local.x*alpha_x * v_local.x*alpha_x) + (v_local.y*alpha_y*v_local.y*alpha_y)) / (v_local.z * v_local.z)) - 1) / 2;
    return 1 / (1 + lambda);
}

/// See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
/// https://jcgt.org/published/0007/04/01/
inline Vector3 sample_visible_normals(const Vector3 &local_dir_in, Real alpha, const Vector2 &rnd_param) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        return -sample_visible_normals(-local_dir_in, alpha, rnd_param);
    }

    // Transform the incoming direction to the "hemisphere configuration".
    Vector3 hemi_dir_in = normalize(
        Vector3{alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z});

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    Real r = sqrt(rnd_param.x);
    Real phi = 2 * c_PI * rnd_param.y;
    Real t1 = r * cos(phi);
    Real t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    Real s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    Vector3 disk_N{t1, t2, sqrt(max(Real(0), 1 - t1*t1 - t2*t2))};

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    Frame hemi_frame(hemi_dir_in);
    Vector3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return normalize(Vector3{alpha * hemi_N.x, alpha * hemi_N.y, max(Real(0), hemi_N.z)});
}

inline Vector3 sample_visible_normals(const Vector3 &local_dir_in, Real alpha_x, Real alpha_y, const Vector2 &rnd_param) {
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        return -sample_visible_normals(-local_dir_in, alpha_x, alpha_y, rnd_param);
    }

    // Transform the incoming direction to the "hemisphere configuration".
    Vector3 hemi_dir_in = normalize(
        Vector3{alpha_x * local_dir_in.x, alpha_y * local_dir_in.y, local_dir_in.z});

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    Real r = sqrt(rnd_param.x);
    Real phi = 2 * c_PI * rnd_param.y;
    Real t1 = r * cos(phi);
    Real t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    Real s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    Vector3 disk_N{t1, t2, sqrt(max(Real(0), 1 - t1*t1 - t2*t2))};

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    Frame hemi_frame(hemi_dir_in);
    Vector3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return normalize(Vector3{alpha_x * hemi_N.x, alpha_y * hemi_N.y, max(Real(0), hemi_N.z)});
}

inline Real sqr(Real x) { return x * x; }

inline Vector2 sqr(Vector2 v) { return Vector2(v.x * v.x, v.y * v.y); }

inline Real saturate(Real x) { return std::clamp(x, Real(0), Real(1)); }

inline Spectrum saturate(Spectrum v) { return Spectrum(saturate(v[0]), saturate(v[1]), saturate(v[2])); }

inline Real depol(Vector2 polV){ return 0.5 * (polV.x + polV.y); }

inline Spectrum depolColor(Spectrum colS, Spectrum colP){ return 0.5 * (colS + colP); }

inline Real smithG1_GGX(Real NdotV, Real alpha) {
    Real a2 = sqr(alpha);
    return 2.0 / (1.0 + sqrt(1.0 + a2 * (1.0 - sqr(NdotV)) / sqr(NdotV)));
}

inline Real smithG_GGX(Real NdotV, Real NdotL, Real alpha) {
    return smithG1_GGX(NdotV, alpha) * smithG1_GGX(NdotL, alpha);
}

inline std::pair<Vector2, Vector2> F_dielectric(Real n_dot_i, Real eta1, Real eta2) {
    Real eta = eta1 / eta2;
    Real cos1 = n_dot_i;
    Real sin1 = (1 - cos1 * cos1);
    Vector2 R; Vector2 phi;

    if (sqr(eta) * sin1 > 1) {
        // total internal reflection
        R = Vector2(1.0, 1.0);
        Real sqrt_term = sqrt(sin1 - 1.0 / (eta * eta));
        Vector2 var = Vector2f(- eta * eta * sqrt_term / cos1, -sqrt_term / cos1);
        phi = Vector2(2 * atan(var.x), 2 * atan(var.y));
        assert(!isnan(phi.x) && !isnan(phi.y));
        return {R, phi};
    }
    else {
        Real cos2 = sqrt(1 - (eta * eta) * sin1);
        Vector2 r = Vector2(
            (eta2 * cos1 - eta1 * cos2) / (eta2 * cos1 + eta1 * cos2),
            (eta1 * cos1 - eta2 * cos2) / (eta1 * cos1 + eta2 * cos2)
        );

        phi = Vector2(
            r.x < 0 ? c_PI : 0,
            r.y < 0 ? c_PI : 0
        );

        R = Vector2(r.x * r.x, r.y * r.y);
        return {R, phi};
    }
}

inline std::pair<Vector2, Vector2> F_conductor(Real n_dot_i, Real eta1, Real eta2, Real k) {
    if (k == 0) {
        return F_dielectric(n_dot_i, eta1, eta2);
    }

    Real cos1 = n_dot_i;

    Real A = sqr(eta2) * (1.0 - sqr(k)) - sqr(eta1) * (1.0 - sqr(cos1));
    Real B = sqrt(sqr(A) + sqr(2.0 * sqr(eta2) * k));
    Real U = sqrt((A + B) / 2.0);
    Real V = sqrt((B - A) / 2.0);

    Vector2 R;
    Vector2 phi;

    R.y = (sqr(eta1 * cos1 - U) + sqr(V)) / (sqr(eta1 * cos1 + U) + sqr(V));

    Vector2 var1 = Vector2(2.0 * eta1 * V * cos1, sqr(U) + sqr(V) - sqr(eta1 * cos1));
    phi.y = atan2(var1.x, var1.y) + c_PI;

    R.x = (sqr(sqr(eta2)*(1-sqr(k))*cos1 - eta1*U) + sqr(2*sqr(eta2)*k*cos1 - eta1*V)) / (sqr(sqr(eta2)*(1-sqr(k))*cos1 + eta1*U) + sqr(2*sqr(eta2)*k*cos1 + eta1*V));

    Vector2 var2 = Vector2(2*eta1*sqr(eta2)*cos1 * (2*k*U - (1-sqr(k))*V), sqr(sqr(eta2)*(1+sqr(k))*cos1) - sqr(eta1)*(sqr(U)+sqr(V))) ;
    phi.x = atan2(var2.x, var2.y);

    return {R, phi};
}

inline Spectrum eval_sensitivity(Real opd, Real shift) {
    Real phase = 2.0 * c_PI * opd * 1e-6;
    Spectrum val = Spectrum(5.4856e-13, 4.4201e-13, 5.2481e-13);
    Spectrum pos = Spectrum(1.6810e+06, 1.7953e+06, 2.2084e+06);
    Spectrum var = Spectrum(4.3278e+09, 9.3046e+09, 6.6121e+09);
    Spectrum xyz = val * sqrt(2.0 * c_PI * var) * cos(pos * phase + shift) * exp(-var * phase * phase);
    xyz.x += 9.7470e-14 * sqrt(2.0 * c_PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift) * exp(-4.5282e+09 * phase * phase);
    return xyz / 1.0685e-7;
}