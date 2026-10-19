import React, { Suspense, useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    useGLTF,
    Html,
} from "@react-three/drei";
import * as THREE from "three";

/* ============================================================
   PANCREAS MODEL
============================================================ */

function PancreasModel() {
    const { scene } = useGLTF(
        "/models/pancreas/pancreas.glb"
    );

    useEffect(() => {
        /* --------------------------------------------------------
           FIND MODEL BOUNDING BOX
        -------------------------------------------------------- */

        const box = new THREE.Box3().setFromObject(scene);

        const center = new THREE.Vector3();
        const size = new THREE.Vector3();

        box.getCenter(center);
        box.getSize(size);

        const maxDimension = Math.max(
            size.x,
            size.y,
            size.z
        );

        /*
          Target size inside the viewer.
    
          Increasing this value makes the pancreas larger.
        */

        const targetSize = 4.8;

        const scale =
            maxDimension > 0
                ? targetSize / maxDimension
                : 1;

        /* --------------------------------------------------------
           CENTER MODEL
        -------------------------------------------------------- */

        scene.position.set(
            -center.x * scale,
            -center.y * scale,
            -center.z * scale
        );

        /* --------------------------------------------------------
           SCALE MODEL
        -------------------------------------------------------- */

        scene.scale.set(
            scale,
            scale,
            scale
        );

        /* --------------------------------------------------------
           MATERIAL SETTINGS
        -------------------------------------------------------- */

        scene.traverse((child) => {
            if (!child.isMesh) {
                return;
            }

            child.castShadow = true;
            child.receiveShadow = true;

            if (child.material) {
                /*
                  Preserve existing textures.
                */

                if (!child.material.map) {
                    child.material.color =
                        new THREE.Color("#d88b82");
                }

                child.material.roughness = 0.45;
                child.material.metalness = 0;

                /*
                  Better anatomy appearance.
                */

                child.material.side =
                    THREE.DoubleSide;

                child.material.needsUpdate = true;
            }
        });

        console.log(
            "PANCREAS MODEL SIZE:",
            size
        );

        console.log(
            "PANCREAS MODEL SCALE:",
            scale
        );

    }, [scene]);

    return (
        <primitive
            object={scene}
        />
    );
}

/* ============================================================
   LOADING
============================================================ */

function Loading() {
    return (
        <Html center>
            <div
                style={{
                    color: "#ffffff",
                    fontSize: "14px",
                    fontWeight: "600",
                    whiteSpace: "nowrap",
                    background: "rgba(15,21,34,0.75)",
                    padding: "10px 14px",
                    borderRadius: "10px",
                    backdropFilter: "blur(8px)",
                }}
            >
                Loading pancreas...
            </div>
        </Html>
    );
}

/* ============================================================
   PANCREAS 3D
============================================================ */

export default function Pancreas3D() {
    return (
        <div className="pancreas-3d-container">

            <Canvas
                camera={{
                    position: [0, 0, 5.2],
                    fov: 42,
                    near: 0.1,
                    far: 100,
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
                    intensity={1.8}
                />

                <hemisphereLight
                    intensity={2}
                    skyColor="#ffffff"
                    groundColor="#202838"
                />

                <directionalLight
                    position={[5, 7, 6]}
                    intensity={3.5}
                    castShadow
                />

                <directionalLight
                    position={[-5, 4, 3]}
                    intensity={2}
                />

                <directionalLight
                    position={[0, -3, -4]}
                    intensity={1.2}
                />

                <pointLight
                    position={[0, 2, 5]}
                    intensity={1.8}
                />

                {/* ====================================================
            MODEL
        ==================================================== */}

                <Suspense fallback={<Loading />}>

                    <PancreasModel />

                    <Environment
                        preset="studio"
                    />

                </Suspense>

                {/* ====================================================
            CONTROLS
        ==================================================== */}

                <OrbitControls
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                    enableDamping={true}
                    dampingFactor={0.08}
                    minDistance={1.5}
                    maxDistance={12}
                    zoomSpeed={0.8}
                    rotateSpeed={0.7}
                    panSpeed={0.7}
                />

            </Canvas>

            {/* ======================================================
          TITLE
      ====================================================== */}

            <div className="pancreas-3d-title">

                <span>
                    🧬
                </span>

                Interactive Human Pancreas

            </div>

            {/* ======================================================
          CONTROLS HELP
      ====================================================== */}

            <div className="pancreas-3d-help">

                Drag to rotate • Scroll to zoom • Right drag to move

            </div>

        </div>
    );
}

/* ============================================================
   PRELOAD
============================================================ */

useGLTF.preload(
    "/models/pancreas/pancreas.glb"
);