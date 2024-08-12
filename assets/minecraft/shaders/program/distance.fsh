#version 330 core

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;

uniform vec2 ScreenSize;

in mat4 modelViewMat;
in mat4 projMat;
in vec2 texCoord;
out vec4 fragColor;

float near = 0.1; 
float far  = 1000.0;

const mat3 sobel_y = mat3(
	vec3(1.0, 0.0, -1.0),
	vec3(2.0, 0.0, -2.0),
	vec3(1.0, 0.0, -1.0)
);

const mat3 sobel_x = mat3(
	vec3(1.0, 2.0, 1.0),
	vec3(0.0, 0.0, 0.0),
	vec3(-1.0, -2.0, -1.0)
);

float linearDepth(vec2 uv) {
    float depth = texture(DiffuseDepthSampler, uv).r * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - depth * (far - near));
}

vec3 reconstructPosition(vec2 uv, float z, mat4 invVp) {
    float x = uv.x * 2.0 - 1.0;
    float y = (1.0 - uv.y) * 2.0 - 1.0;
    vec4 pos_s = vec4(x,y,z,1.0);
    vec4 pos_v = invVp*pos_s;
    vec4 pos_w = invModelView*pos_v;
    mat4 invModelView = inverse(modelViewMat);
    return pos_w.xyz/pos_w.w;
}

vec3 directions[10] = { 
    vec3( 1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0), // +X, -X
    vec3( 0.0,  1.0,  0.0), vec3( 0.0, -1.0,  0.0), // +Y, -Y
    vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0, -1.0), // +Z, -Z
    // vec3( 1.0,  1.0,  0.0), vec3(-1.0, -1.0,  0.0), // +XY, -XY
    // vec3( 1.0, -1.0,  0.0), vec3(-1.0,  1.0,  0.0), // +X-Y, -X+Y
    vec3( 1.0,  0.0,  1.0), vec3(-1.0,  0.0, -1.0), // +XZ, -XZ
    vec3( 1.0,  0.0, -1.0), vec3(-1.0,  0.0,  1.0), // +X-Z, -X+Z
    // vec3( 0.0,  1.0,  1.0), vec3( 0.0, -1.0, -1.0), // +YZ, -YZ
    // vec3( 0.0,  1.0, -1.0), vec3( 0.0, -1.0,  1.0), // +Y-Z, -Y+Z
    // vec3( 1.0,  1.0,  1.0), vec3(-1.0, -1.0, -1.0), // +XYZ, -XYZ
    // vec3( 1.0,  1.0, -1.0), vec3(-1.0, -1.0,  1.0), // +XY-Z, -XY+Z
    // vec3( 1.0, -1.0,  1.0), vec3(-1.0,  1.0, -1.0), // +X-YZ, -X+YZ
    // vec3( 1.0, -1.0, -1.0), vec3(-1.0,  1.0,  1.0)  // +X-Y-Z, -X+Y+Z
};

vec3 findClosest(vec3 v) {
    float maxDot = -1.0;
    vec3 closestDirection = directions[0];

    for (int i = 0; i < 10; i++) {
        float dotProduct = dot(normalize(v), directions[i]);
        if (dotProduct > maxDot) {
            maxDot = dotProduct;
            closestDirection = directions[i];
        }
    }

    return closestDirection;
}

vec3 calculateNormal(vec2 uv0) {
    mat4 invProj = inverse(projMat);
    vec2 depthDimensions = textureSize(DiffuseDepthSampler, 0);
    
    vec2 uv1 = uv0 + vec2(1.0, 0.0) / depthDimensions;
    vec2 uv2 = uv0 + vec2(0.0, 1.0) / depthDimensions;

    float depth0 = linearDepth(uv0);
    float depth1 = linearDepth(uv1);
    float depth2 = linearDepth(uv2);

    vec3 p0 = reconstructPosition(uv0, depth0, invProj);
    vec3 p1 = reconstructPosition(uv1, depth1, invProj);
    vec3 p2 = reconstructPosition(uv2, depth2, invProj);

    vec3 normal = normalize(cross(p2-p0, p1-p0));
    return findClosest(normal);
}

vec3 smoothNormal(vec2 uv0) {
    vec2 offset = 1.0 / ScreenSize;
    vec3 normal = calculateNormal(uv0);

    vec3 n  = calculateNormal(uv0 + vec2( 0.0     , -offset.y));
    vec3 s  = calculateNormal(uv0 + vec2( 0.0     ,  offset.y));
    vec3 e  = calculateNormal(uv0 + vec2( offset.x,  0.0     ));
    vec3 w  = calculateNormal(uv0 + vec2(-offset.x,  0.0     ));
    vec3 nw = calculateNormal(uv0 + vec2(-offset.x, -offset.y));
    vec3 ne = calculateNormal(uv0 + vec2( offset.x, -offset.y));
    vec3 sw = calculateNormal(uv0 + vec2(-offset.x,  offset.y));
    vec3 se = calculateNormal(uv0 + vec2( offset.x,  offset.y));

    vec3 averageNormal = normalize(
        normal + n + s + e + w + nw + ne + sw + se
    );
    return findClosest(averageNormal);
}

vec3 dominantNormal(vec2 uv) {
    vec2 offset = 1.0 / ScreenSize;
    vec3 normal = calculateNormal(uv);

    vec3 n  = calculateNormal(uv + vec2( 0.0     , -offset.y));
    vec3 s  = calculateNormal(uv + vec2( 0.0     ,  offset.y));
    vec3 e  = calculateNormal(uv + vec2( offset.x,  0.0     ));
    vec3 w  = calculateNormal(uv + vec2(-offset.x,  0.0     ));
    vec3 nw = calculateNormal(uv + vec2(-offset.x, -offset.y));
    vec3 ne = calculateNormal(uv + vec2( offset.x, -offset.y));
    vec3 sw = calculateNormal(uv + vec2(-offset.x,  offset.y));
    vec3 se = calculateNormal(uv + vec2( offset.x,  offset.y));

    vec3 normals[9];

    int counts[10];

    for (int i = 0; i < 9; i++) {
        vec3 closest = findClosest(normals[i]);
        for (int j = 0; j < 10; j++) {
            if (closest == directions[j]) {
                counts[j]++;
                break;
            }
        }
    }

    int maxCount = 0;
    int dominantIndex = 0;
    for (int i = 0; i < 10; i++) {
        if (counts[i] > maxCount) {
            maxCount = counts[i];
            dominantIndex = i;
        }
    }

    return directions[dominantIndex];
}

vec3 calculateNormalCross(vec2 uv0) {
    mat4 invProj = inverse(projMat);
    vec2 depthDimensions = textureSize(DiffuseDepthSampler, 0);
    
    vec2 uvLeft = uv0 + vec2(-1.0, 0.0) / depthDimensions;
    vec2 uvRight = uv0 + vec2(1.0, 0.0) / depthDimensions;
    vec2 uvUp = uv0 + vec2(0.0, 1.0) / depthDimensions;
    vec2 uvDown = uv0 + vec2(0.0, -1.0) / depthDimensions;

    float depthCenter = linearDepth(uv0);
    float depthLeft = linearDepth(uvLeft);
    float depthRight = linearDepth(uvRight);
    float depthUp = linearDepth(uvUp);
    float depthDown = linearDepth(uvDown);

    vec3 pCenter = reconstructPosition(uv0, depthCenter, invProj);
    vec3 pLeft = reconstructPosition(uvLeft, depthLeft, invProj);
    vec3 pRight = reconstructPosition(uvRight, depthRight, invProj);
    vec3 pUp = reconstructPosition(uvUp, depthUp, invProj);
    vec3 pDown = reconstructPosition(uvDown, depthDown, invProj);

    vec3 p1 = (depthLeft < depthRight) ? pLeft : pRight;
    vec3 p2 = (depthUp < depthDown) ? pUp : pDown;

    vec3 normal = normalize(cross(p2 - pCenter, p1 - pCenter));
    return findClosest(normal);
}

#define DEBUG

void main() {
    vec2 uv = texCoord;
    
    vec2 offset = 1.0 / ScreenSize;
    float depth = linearDepth(uv);

#ifdef DEBUG
    if (uv.x > 0.5) uv.x -= 0.5;
    if (uv.y > 0.5) uv.y -= 0.5;
    if (texCoord.x > 0.5 && texCoord.y > 0.5) {
        vec3 normal = calculateNormalCross(uv);

        vec3 n  = calculateNormalCross(uv + vec2( 0.0     , -offset.y));
        vec3 s  = calculateNormalCross(uv + vec2( 0.0     ,  offset.y));
        vec3 e  = calculateNormalCross(uv + vec2( offset.x,  0.0     ));
        vec3 w  = calculateNormalCross(uv + vec2(-offset.x,  0.0     ));
        vec3 nw = calculateNormalCross(uv + vec2(-offset.x, -offset.y));
        vec3 ne = calculateNormalCross(uv + vec2( offset.x, -offset.y));
        vec3 sw = calculateNormalCross(uv + vec2(-offset.x,  offset.y));
        vec3 se = calculateNormalCross(uv + vec2( offset.x,  offset.y));

        mat3 surrounding_pixels = mat3(
            vec3(length(nw-normal), length(n-normal), length(ne-normal)),
            vec3(length(w-normal), length(normal-normal), length(e-normal)),
            vec3(length(sw-normal), length(s-normal), length(se-normal))
        );

        float edge_x = dot(sobel_x[0], surrounding_pixels[0]) + dot(sobel_x[1], surrounding_pixels[1]) + dot(sobel_x[2], surrounding_pixels[2]);
        float edge_y = dot(sobel_y[0], surrounding_pixels[0]) + dot(sobel_y[1], surrounding_pixels[1]) + dot(sobel_y[2], surrounding_pixels[2]);

        float edge = sqrt(pow(edge_x, 2.0)+pow(edge_y, 2.0));
        if (edge > (linearDepth(uv) > 20.0 ? 5.0 : 2.0)) {
            fragColor = vec4(vec3(edge_x, edge_y, 0), 1.0);
        } else {
            fragColor = texture(DiffuseSampler, uv);//vec4(normal, 1.0);
        }
    } else if (texCoord.y > 0.5 && texCoord.x <= 0.5) {
        fragColor = vec4(vec3(linearDepth(uv)/500), 1.0);
    } else if (texCoord.y <= 0.5 && texCoord.x <= 0.5) {
        vec3 normal = calculateNormalCross(uv);
        fragColor = vec4(normal, 1.0);
    } else if (texCoord.y <= 0.5 && texCoord.x > 0.5) {
        fragColor = texture(DiffuseSampler, uv);
    }
#else
    vec3 normal = calculateNormalCross(uv);

    vec3 n  = calculateNormalCross(uv + vec2( 0.0     , -offset.y));
    vec3 s  = calculateNormalCross(uv + vec2( 0.0     ,  offset.y));
    vec3 e  = calculateNormalCross(uv + vec2( offset.x,  0.0     ));
    vec3 w  = calculateNormalCross(uv + vec2(-offset.x,  0.0     ));
    vec3 nw = calculateNormalCross(uv + vec2(-offset.x, -offset.y));
    vec3 ne = calculateNormalCross(uv + vec2( offset.x, -offset.y));
    vec3 sw = calculateNormalCross(uv + vec2(-offset.x,  offset.y));
    vec3 se = calculateNormalCross(uv + vec2( offset.x,  offset.y));

    mat3 surrounding_pixels = mat3(
        vec3(length(nw-normal), length(n-normal), length(ne-normal)),
        vec3(length(w-normal), length(normal-normal), length(e-normal)),
        vec3(length(sw-normal), length(s-normal), length(se-normal))
    );

    float edge_x = dot(sobel_x[0], surrounding_pixels[0]) + dot(sobel_x[1], surrounding_pixels[1]) + dot(sobel_x[2], surrounding_pixels[2]);
    float edge_y = dot(sobel_y[0], surrounding_pixels[0]) + dot(sobel_y[1], surrounding_pixels[1]) + dot(sobel_y[2], surrounding_pixels[2]);

    float edge = sqrt(pow(edge_x, 2.0)+pow(edge_y, 2.0));
    if (edge > (linearDepth(uv) > 20.0 ? 5.0 : 2.0)) {
        fragColor = vec4(vec3(edge_x, edge_y, 0), 1.0);
    } else {
        fragColor = texture(DiffuseSampler, uv);//vec4(normal, 1.0);
    }
#endif
}