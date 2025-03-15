"use client"

import React from 'react';
import {Panel, PanelGroup, PanelResizeHandle} from 'react-resizable-panels';
import {Box} from "@mui/material";
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';


interface ResizableProps {
    isOpen: boolean;
    onClose: () => void;
    children: React.ReactNode;
    loading: boolean;
}

const ResizableDialog = React.memo( ({isOpen, onClose, children, loading} : ResizableProps) => {

    return (
        isOpen && (
            loading? (
                <Box className="fixed top-0 left-0 w-full h-full bg-gray-900 bg-opacity-50 z-50 flex justify-center items-center">
                    <Box className="animate-spin rounded-full border-4 border-t-white border-r-white border-b-white border-l-white h-32 w-32">

                    </Box>
                </Box>
            ) :
            (
                <PanelGroup className={`fixed h-full right-0 left-0 top-0 z-20 bg-transparent pointer-events-none opacity-80`} direction="vertical">
                    <Panel defaultSize={200} className={`bg-transparent pointer-events-none`}/>
                    <PanelResizeHandle />
                    <Panel defaultSize={180} className="relative pointer-events-auto rounded bg-white px-4 pb-2 pt-0 border-t-2">
                        <Box className="flex flex-col justify-between items-center h-full">
                            <KeyboardArrowDownIcon className={`cursor-pointer`} color="primary" onClick={onClose} />
                            {children}
                        </Box>
                    </Panel>
                </PanelGroup>
            )
       )
    );
})

export default ResizableDialog;