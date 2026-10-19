import React, { Suspense, useEffect, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import {
    OrbitControls,
    Environment,
    Html,
    useGLTF,
} from "@react-three/drei";
import * as THREE from "three";

function LoadingHeart() {
    return (
        <Html center>
            <div
                style={{
                    color: "white",
                    textAlign: "center",
                    fontFamily: "Arial, sans-serif",
                    background: "rgba(7, 16, 30, 0.88)",
                    padding: "18px 24px",
                    borderRadius: "14px",
                    minWidth: "190px",
                }}
            >
                <div style={{ fontSize: "30px", marginBottom: "8px" }}>
                    ❤️
                </div>

                <strong>Loading Heart...</strong>

                <div
                    style={{
                        marginTop: "6px",
                        fontSize: "11px",
                        color: "#aebbd0",
                    }}
                >
                    Loading anatomical 3D model
                </div>
            </div>
        </Html>
    );
}

function HeartModel() {
    const groupRef = useRef();

    const { scene } = useGLTF("/models/heart/heart.glb");

    useEffect(() => {
        if (!scene) return;

        scene.traverse((object) => {
            if (!object.isMesh) return;

            object.castShadow = true;
            object.receiveShadow = true;

            if (object.material) {
                object.material.side = THREE.DoubleSide;
                object.material.roughness = 0.65;
                object.material.metalness = 0.02;
            }
        });
    }, [scene]);

    return (
        <group
            ref={groupRef}
            rotation={[0, 0, 0]}
        >
            <primitive
                object={scene}
                scale={1}
            />
        </group>
    );
}

function HeartScene() {
    const { camera } = useThree();

    useEffect(() => {
        camera.position.set(0, 0.2, 4.8);
        camera.lookAt(0, 0, 0);
    }, [camera]);

    return (
        <>
            {/* Main lighting */}
            <ambientLight intensity={1.7} />

            <directionalLight
                position={[5, 6, 7]}
                intensity={2.4}
                castShadow
            />

            <directionalLight
                position={[-5, 2, 4]}
                intensity={1.3}
            />

            <pointLight
                position={[0, 2, 5]}
                intensity={1.4}
            />

            {/* Heart model */}
            <Suspense fallback={<LoadingHeart />}>
                <HeartModel />
            </Suspense>

            {/* Mouse controls */}
            <OrbitControls
                enableRotate
                enableZoom
                enablePan

                minDistance={2}
                maxDistance={9}

                enableDamping
                dampingFactor={0.08}

                rotateSpeed={0.7}
                zoomSpeed={0.8}
                panSpeed={0.7}
            />

            {/* Studio lighting */}
            <Environment preset="studio" />
        </>
    );
}

export default function Heart3D() {
    return (
        <div className="heart-3d-container">

            <Canvas
                camera={{
                    position: [0, 0.2, 4.8],
                    fov: 45,
                    near: 0.01,
                    far: 100,
                }}

                dpr={[1, 2]}

                gl={{
                    antialias: true,
                    alpha: true,
                    powerPreference: "high-performance",
                }}

                shadows
            >

                <HeartScene />

            </Canvas>

            {/* Overlay */}
            <div className="heart-3d-controls">

                <div className="heart-3d-title">
                    ❤️ Interactive Human Heart
                </div>

                <div className="heart-3d-help">
                    Drag to rotate • Scroll to zoom • Right-click to pan
                </div>

                <div className="heart-3d-source">
                    Anatomical reference model • NIH 3D
                </div>

            </div>

        </div>
    );
}

useGLTF.preload("/models/heart/heart.glb");