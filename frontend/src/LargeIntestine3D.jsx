import React, { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

/* ============================================================
   LARGE INTESTINE MODEL
============================================================ */

function LargeIntestineModel() {
    const { scene } = useGLTF(
        "/models/large_intestine/large_intestine.glb"
    );

    const preparedScene = useMemo(() => {
        const clonedScene = scene.clone(true);

        /* --------------------------------------------------------
           Calculate model bounds
        -------------------------------------------------------- */

        const box = new THREE.Box3().setFromObject(
            clonedScene
        );

        const center = new THREE.Vector3();
        const size = new THREE.Vector3();

        box.getCenter(center);
        box.getSize(size);

        /* --------------------------------------------------------
           Center model
        -------------------------------------------------------- */

        clonedScene.position.sub(center);

        /* --------------------------------------------------------
           Scale model
        -------------------------------------------------------- */

        const maxDimension = Math.max(
            size.x,
            size.y,
            size.z
        );

        const targetSize = 5.2;

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
           Improve materials
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
                  Preserve the original texture
                  when one exists.
                */

                if (!material.map) {
                    material.color =
                        new THREE.Color("#c9787d");
                }

                material.roughness = 0.5;
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
   LARGE INTESTINE 3D
============================================================ */

function LargeIntestine3D() {
    return (
        <div className="large-intestine-3d-container">

            <Canvas
                camera={{
                    position: [0, 0.2, 5.8],
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
            LARGE INTESTINE
        ---------------------------------------------------- */}

                <LargeIntestineModel />

                {/* ----------------------------------------------------
            ORBIT CONTROLS
        ---------------------------------------------------- */}

                <OrbitControls
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    minDistance={2.5}
                    maxDistance={9}
                    target={[0, 0, 0]}
                />

            </Canvas>

            {/* ======================================================
          TITLE
      ====================================================== */}

            <div className="large-intestine-3d-title">
                🧬 Interactive Large Intestine
            </div>

            {/* ======================================================
          HELP
      ====================================================== */}

            <div className="large-intestine-3d-help">
                Drag to rotate • Scroll to zoom • Right-click to pan
            </div>

        </div>
    );
}

/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    "/models/large_intestine/large_intestine.glb"
);

export default LargeIntestine3D;