"use client";

import React, { useState, useEffect } from "react";
import {
    Container,
    Typography,
    Table,
    TableHead,
    TableRow,
    TableCell,
    TableBody,
    TextField,
    Button,
    FormControlLabel,
    Checkbox,
    TableContainer, Paper
} from "@mui/material";
import InputFileUpload from "@/app/components/InputFileUpload";
import {err, success} from "@/app/utils/alerter";
import {get} from "@/app/utils/api";
import {MyResponse, ROLE_TYPE} from "@/app/types/common";
import {convertToChinaTime, handleDownload} from "@/app/utils/common";
import { Project } from '@/app/types/das'
import {navigateTo} from "@/app/utils/navigator";
import {analysisPublicProject, getPublicList} from "@/app/api/das";
import {useAuth} from "@/app/context/AuthContext";

export default function Page() {
    /**
     * 获取公共项目列表
     */
    useEffect(() => {
        getProjects();
    }, []);
    const [projects, setProjects] = useState<Project[]>([]);
    const getProjects = async () => {
        try {
            const res: MyResponse = await getPublicList()
            if (res.success){
                for (let i = 0; i < res.data.length; i++) {
                    res.data[i].createdTime = convertToChinaTime(res.data[i].createdTime);
                }
                setProjects(res.data);
            }else{
                err(res.message);
            }
        }catch (e){
            console.error(e);
            err("Failed to get projects.");
        }
    }
    const analysisProject = (project: Project) => {
        navigateTo('/pages/dashboard/project-analysis', {
            jsonPath: project.jsonPath as string,
            projectName: project.projectName,
            analysised: project.analysised || 0,
            projectId: project.id
        })
    }

    /**
     * 新增项目
     */
    const [newProject, setNewProject] = useState<Project>({
        projectName: "",
        isPublic: true,
        dwgPath: "",
        id: 0,
        analysised: 0
    });

    const roleContext = useAuth()

    const handleFileChange = (files: string[]) => {
        setNewProject({...newProject, dwgPath: files[0]});
    };

    const handleSubmit = async () => {
        if (!newProject.isPublic){
            err("Only public projects are allowed currently.")
            return;
        }
        if (!newProject.dwgPath) {
            err("Please upload a file.");
            return;
        }
        if (!newProject.isPublic) {
            err("Only public projects are allowed currently.");
            return;
        }
        analysisPublicProject(
            newProject.projectName,
            newProject.dwgPath,
            newProject.isPublic ? 1 : 0
        ).then(res=>{
            if (res.success){
                resetNewProject()
                getProjects();
                success("Project created successfully.");
            }else{
                err(res.message);
            }
        }).catch(err=>{
            console.error(err);
            err("Failed to create project.");
        })

    };

    const [uploadKey, setUploadKey] = useState(0);
    const resetNewProject = () => {
        setNewProject({
            projectName: "",
            isPublic: true,
            dwgPath: "",
            id: 0,
            analysised: 0
        });
        setUploadKey(uploadKey + 1);
    };

    return (
        <div className="bg-gray-100">
            <Container
                maxWidth="md"
                className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4 relative"
            >
                {/* Public Projects List */}
                <div className="w-full !mb-8">
                    <Typography variant="h5" className="text-center !mb-4 text-gray-700">
                        Public Demo Projects
                    </Typography>
                    <TableContainer component={Paper} className="w-full max-h-[50vh]">
                        <Table stickyHeader className="bg-white shadow-md rounded-lg">
                            <TableHead>
                                <TableRow>
                                    <TableCell align={`center`}>Project Name</TableCell>
                                    <TableCell align={`center`}>Created Time</TableCell>
                                    <TableCell align={`center`}>DWG Path</TableCell>
                                    <TableCell align={`center`}>DESIGN</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {projects.map((project, index) => (
                                    <TableRow key={index}>
                                        <TableCell align={`center`}>{project.projectName}</TableCell>
                                        <TableCell align={`center`}>{project.createdTime}</TableCell>
                                        <TableCell align={`center`}>
                                            <Button onClick={()=>handleDownload(project.dwgPath)}>download</Button>
                                        </TableCell>
                                        <TableCell align={`center`}>
                                            <Button variant="contained" onClick={()=>analysisProject(project)}>ANALYSIS</Button>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </div>

                {/* Add New Project */}
                <div className="w-full bg-white p-6 shadow-md rounded-lg">
                    <Typography variant="h5" className="text-center !mb-4 text-gray-700">
                        Add New Public Project
                    </Typography>
                    <div className="!mb-4">
                        <TextField
                            fullWidth
                            label="Project Name"
                            value={newProject.projectName}
                            onChange={(e) => setNewProject({...newProject, projectName: e.target.value })}
                            className="!mb-4"
                        />
                        <div className="flex items-center justify-between">
                            <InputFileUpload
                                disabled={!roleContext.isSuperManager}
                                apiUrl="/file-manage/dwg/upload"
                                maxFiles={1}
                                acceptTypes=".dwg"
                                onSuccess={handleFileChange}  // 上传成功后的回调函数
                                key={uploadKey} // 提交后重置组件
                            />
                            <FormControlLabel
                                control={
                                    // <Checkbox
                                    //     checked={newProject.isPublic}
                                    //     onChange={(e) => setNewProject({...newProject, isPublic: e.target.checked })}
                                    //     color="primary"
                                    // />
                                    <Checkbox
                                        checked={true}
                                        color="primary"
                                    />
                                }
                                label="Is Public"
                                sx={{
                                    '& .MuiFormControlLabel-label': {
                                        color: '#555',  // Tailwind's gray-500 color
                                    },
                                }}
                            />
                        </div>
                    </div>
                    <Button
                        variant="contained"
                        color="primary"
                        onClick={handleSubmit}
                        className="w-full"
                        disabled={!roleContext.isSuperManager}
                    >
                        Submit
                    </Button>
                </div>
            </Container>
        </div>
    );
}
