"use client"

import React from "react";
import {Box, Button} from "@mui/material";
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

type TotalControlPanelProps = {
    projectName: string;
    handleShowStdChange: () => void;
    handleShowCurComChange: () => void;
    handleShowRelayChange: () => void;

};

const TotalControlPanel: React.FC<TotalControlPanelProps> = ({
    projectName,
    handleShowCurComChange,
    handleShowStdChange,
    handleShowRelayChange
                                                             }) => {
    const [expanded, setExpanded] = React.useState<boolean>(false)
    const handleExpandChange = () => {
        setExpanded(!expanded)
    }

    return (
        <div className={`${expanded? "opacity-100" : "opacity-80 h-6"} rounded transition-all duration-300 ease-in-out fixed top-0 left-1/2 -translate-x-1/2 bg-white shadow-md p-2 flex justify-center gap-4 z-50`}>
            <Box
                className="flex items-center cursor-pointer"
                onClick={handleExpandChange}
            >
                {expanded ? <KeyboardArrowUpIcon color="primary"/> : <KeyboardArrowDownIcon color="primary" />}
            </Box>
            {expanded && (
                <>
                    <h2 className="text-lg font-bold text-blue-600 mx-2 my-auto w-36 text-center overflow-hidden whitespace-nowrap text-ellipsis">{projectName}</h2>
                    <Button variant="contained" color="success" onClick={handleShowRelayChange}>Graphical Structure</Button>
                    <Button variant="contained" color="primary" onClick={handleShowCurComChange}>Current Components</Button>
                    <Button variant="contained" color="secondary" onClick={handleShowStdChange}>Standard Components</Button>
                </>
            )}
        </div>
    );
};

export default TotalControlPanel;
