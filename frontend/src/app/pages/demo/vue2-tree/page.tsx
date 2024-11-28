"use client"

import React, { useState } from "react";
import { Box, Button, TextField, Typography, Paper } from "@mui/material";
import { styled } from "@mui/system";

const Container = styled(Box)({
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    backgroundColor: "#f5f5f5",
    gap: "16px",
});

const Wheel = styled(Box)({
    width: "300px",
    height: "300px",
    borderRadius: "50%",
    border: "5px solid #1976d2",
    position: "relative",
    overflow: "hidden",
});

const Segment = styled(Box)(({ rotation }: { rotation: number }) => ({
    position: "absolute",
    top: "50%",
    left: "50%",
    transformOrigin: "0% 0%",
    transform: `rotate(${rotation}deg)`,
    width: "50%",
    height: "50%",
    backgroundColor: `hsl(${rotation % 360}, 70%, 70%)`,
    clipPath: "polygon(0 0, 100% 0, 50% 100%)",
}));

const Pointer = styled(Box)({
    position: "absolute",
    top: "-10px",
    left: "calc(50% - 10px)",
    width: "20px",
    height: "20px",
    backgroundColor: "#ff5722",
    clipPath: "polygon(50% 0, 100% 100%, 0% 100%)",
    zIndex: 10,
});

const EatWheel: React.FC = () => {
    const [options, setOptions] = useState<string[]>(["寿司", "火锅", "意面", "烤肉"]);
    const [newOption, setNewOption] = useState<string>("");
    const [selected, setSelected] = useState<string | null>(null);
    const [rotation, setRotation] = useState<number>(0);

    const handleAddOption = () => {
        if (newOption.trim() && !options.includes(newOption)) {
            setOptions([...options, newOption]);
            setNewOption("");
        }
    };

    const spinWheel = () => {
        const randomIndex = Math.floor((Date.now() % options.length) + Math.random() * options.length) % options.length;
        const randomDegree = 360 / options.length;
        const newRotation = rotation + 360 * 3 + randomIndex * randomDegree;
        setRotation(newRotation);
        setTimeout(() => {
            setSelected(options[randomIndex]);
        }, 3000);
    };

    return (
        <Container>
            <Typography variant="h4" gutterBottom>
                今天吃什么？
            </Typography>
            <Wheel style={{ transform: `rotate(${rotation}deg)`, transition: "transform 3s ease-out" }}>
                {options.map((option, index) => (
                    <Segment key={index} rotation={(index * 360) / options.length}>
                        <Typography
                            variant="caption"
                            style={{
                                position: "absolute",
                                top: "75%",
                                left: "50%",
                                transform: "translate(-50%, -50%) rotate(-90deg)",
                                fontSize: "14px",
                                fontWeight: "bold",
                            }}
                        >
                            <span className="ml-10">{option}</span>
                        </Typography>
                    </Segment>
                ))}
            </Wheel>
            <Pointer />
            {selected && (
                <Paper
                    elevation={3}
                    style={{
                        padding: "8px 16px",
                        marginTop: "16px",
                        backgroundColor: "#ffe0b2",
                        fontWeight: "bold",
                    }}
                >
                    <Typography>{`今天吃：${selected}`}</Typography>
                </Paper>
            )}
            <Box display="flex" gap="8px">
                <TextField
                    variant="outlined"
                    size="small"
                    value={newOption}
                    onChange={(e) => setNewOption(e.target.value)}
                    placeholder="添加选项"
                />
                <Button variant="contained" color="primary" onClick={handleAddOption}>
                    添加
                </Button>
            </Box>
            <Button variant="contained" color="secondary" onClick={spinWheel}>
                转！
            </Button>
        </Container>
    );
};

export default EatWheel;
