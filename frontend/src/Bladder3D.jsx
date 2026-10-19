import React, { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

/* ============================================================
   BLADDER MODEL
============================================================ */

function BladderModel() {
    const { scene } = useGLTF(
        "/models/bladder/bladder.glb"
    );

    const preparedScene = useMemo(() => {
        const clonedScene = scene.clone(true);

        /* --------------------------------------------------------
           CALCULATE MODEL SIZE
        -------------------------------------------------------- */

        const box = new THREE.Box3().setFromObject(
            clonedScene
        );

        const center = new THREE.Vector3();
        const size = new THREE.Vector3();

        box.getCenter(center);
        box.getSize(size);

        /* --------------------------------------------------------
           CENTER MODEL
        -------------------------------------------------------- */

        clonedScene.position.sub(center);

        /* --------------------------------------------------------
           SCALE MODEL
        -------------------------------------------------------- */

        const maxDimension = Math.max(
            size.x,
            size.y,
            size.z
        );

        const targetSize = 4.6;

        if (maxDimension > 0) {
            const scale =
                targetSize / maxDimension;

            clonedScene.scale.set(
                scale,
                scale,
                scale
            );
        }

        /* --------------------------------------------------------
           MATERIAL PREPARATION
        -------------------------------------------------------- */

        clonedScene.traverse((child) => {
            if (!child.isMesh) {
                return;
            }

            child.castShadow = true;
            child.receiveShadow = true;

            if (child.material) {
                const material =
                    child.material.clone();

                /*
                 * Preserve existing textures.
                 * Only apply fallback color when
                 * the model has no texture.
                 */

                if (!material.map) {
                    material.color =
                        new THREE.Color("#c97882");
                }

                material.roughness = 0.48;
                material.metalness = 0.0;
                material.side =
                    THREE.DoubleSide;

                child.material = material;
            }
        });

        return clonedScene;
    }, [scene]);

    return (
        <primitive
            object={preparedScene}
        />
    );
}

/* ============================================================
   BLADDER 3D VIEWER
============================================================ */

function Bladder3D() {
    return (
        <div className="bladder-3d-container">

            <Canvas
                camera={{
                    position: [0, 0.2, 5.5],
                    fov: 42,
                }}
                shadows
                dpr={[1, 2]}
            >

                {/* ----------------------------------------------------
            BACKGROUND
        ---------------------------------------------------- */}

                <color
                    attach="background"
                    args={["#101522"]}
                />

                {/* ----------------------------------------------------
            ENVIRONMENT
        ---------------------------------------------------- */}

                <Environment
                    preset="studio"
                />

                {/* ----------------------------------------------------
            LIGHTING
        ---------------------------------------------------- */}

                <ambientLight
                    intensity={1.2}
                />

                <hemisphereLight
                    skyColor="#ffffff"
                    groundColor="#25304a"
                    intensity={1.3}
                />

                <directionalLight
                    position={[4, 6, 5]}
                    intensity={2.2}
                    castShadow
                />

                <directionalLight
                    position={[-4, 2, 3]}
                    intensity={1.3}
                />

                <pointLight
                    position={[0, 1, 3]}
                    intensity={1.1}
                />

                <pointLight
                    position={[0, -2, -2]}
                    intensity={0.8}
                />

                {/* ----------------------------------------------------
            MODEL
        ---------------------------------------------------- */}

                <BladderModel />

                {/* ----------------------------------------------------
            CONTROLS
        ---------------------------------------------------- */}

                <OrbitControls
                    enablePan
                    enableZoom
                    enableRotate
                    minDistance={2.5}
                    maxDistance={9}
                    target={[0, 0, 0]}
                />

            </Canvas>

            {/* ------------------------------------------------------
          TITLE
      ------------------------------------------------------ */}

            <div className="bladder-3d-title">
                🫧 Interactive Urinary Bladder
            </div>

            {/* ------------------------------------------------------
          HELP
      ------------------------------------------------------ */}

            <div className="bladder-3d-help">
                Drag to rotate • Scroll to zoom • Right-click to pan
            </div>

        </div>
    );
}

/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    "/models/bladder/bladder.glb"
);

export default Bladder3D;