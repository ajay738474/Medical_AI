import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    Bounds,
    useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

/* ============================================================
   EYE MODEL
============================================================ */

function EyeModel() {
    const { scene } = useGLTF(
        "/models/eye/eye.glb"
    );

    /*
     * Clone the model so we can safely modify
     * materials without changing the cached GLTF.
     */
    const model = scene.clone(true);

    let meshCount = 0;

    model.traverse((child) => {

        if (!child.isMesh) {
            return;
        }

        meshCount++;

        child.visible = true;
        child.castShadow = true;
        child.receiveShadow = true;

        /*
         * Clone material.
         */
        if (child.material) {

            const material =
                child.material.clone();

            /*
             * Preserve the original texture.
             * If there is no texture, use a visible
             * anatomical fallback.
             */
            if (!material.map) {

                material.color =
                    new THREE.Color("#d99090");
            }

            material.roughness = 0.35;
            material.metalness = 0;

            material.side =
                THREE.DoubleSide;

            material.transparent = false;
            material.opacity = 1;

            child.material = material;
        }
    });

    console.log(
        "EYE MESH COUNT:",
        meshCount
    );

    return (
        <primitive
            object={model}
        />
    );
}

/* ============================================================
   LOADING
============================================================ */

function EyeLoading() {

    return (
        <mesh>

            <sphereGeometry
                args={[1.5, 32, 32]}
            />

            <meshStandardMaterial
                color="#d98f8f"
                roughness={0.4}
            />

        </mesh>
    );
}

/* ============================================================
   EYE 3D VIEWER
============================================================ */

function Eye3D() {

    return (
        <div className="eye-3d-container">

            <Canvas
                camera={{
                    position: [0, 0, 5],
                    fov: 40,
                    near: 0.001,
                    far: 1000,
                }}

                shadows

                dpr={[1, 2]}
            >

                {/* ====================================================
            BACKGROUND
        ==================================================== */}

                <color
                    attach="background"
                    args={["#101522"]}
                />

                {/* ====================================================
            LIGHTING
        ==================================================== */}

                <ambientLight
                    intensity={2}
                />

                <hemisphereLight
                    skyColor="#ffffff"
                    groundColor="#303b55"
                    intensity={2}
                />

                <directionalLight
                    position={[5, 6, 5]}
                    intensity={3}
                    castShadow
                />

                <directionalLight
                    position={[-5, 3, 4]}
                    intensity={2}
                />

                <directionalLight
                    position={[0, -5, 3]}
                    intensity={1.5}
                />

                <pointLight
                    position={[0, 0, 5]}
                    intensity={2}
                />

                {/* ====================================================
            ENVIRONMENT
        ==================================================== */}

                <Environment
                    preset="studio"
                />

                {/* ====================================================
            AUTOMATIC MODEL FRAMING
        ==================================================== */}

                <Suspense
                    fallback={<EyeLoading />}
                >

                    <Bounds
                        fit
                        clip
                        observe
                        margin={1.3}
                    >

                        <EyeModel />

                    </Bounds>

                </Suspense>

                {/* ====================================================
            CONTROLS
        ==================================================== */}

                <OrbitControls
                    makeDefault

                    enableRotate
                    enableZoom
                    enablePan

                    minDistance={0.5}
                    maxDistance={20}

                    dampingFactor={0.08}
                    enableDamping
                />

            </Canvas>

            {/* ======================================================
          TITLE
      ====================================================== */}

            <div className="eye-3d-title">
                👁️ Interactive Human Eye
            </div>

            {/* ======================================================
          HELP
      ====================================================== */}

            <div className="eye-3d-help">
                Drag to rotate • Scroll to zoom • Right-click to pan
            </div>

        </div>
    );
}

/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    "/models/eye/eye.glb"
);

export default Eye3D;