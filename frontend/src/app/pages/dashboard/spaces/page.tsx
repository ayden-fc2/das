"use client"

import React, {useEffect} from "react";
import {
    Accordion, AccordionDetails,
    AccordionSummary,
    Box,
    Button, Dialog, DialogActions,
    DialogContent,
    DialogTitle,
    Divider,
    MenuItem,
    Select,
    SelectChangeEvent, TextField,
    Typography
} from "@mui/material";
import {getAllMyOrgs} from "@/app/api/org";
import {err} from "@/app/utils/alerter";
import {createProjectApi, getAllProjectsApi} from "@/app/api/project";
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {convertToChinaTime, handleCopy} from "@/app/utils/common";

interface Org {
    org_id: string;
    org_name: string;
    authorityIds: string
}

const Spaces: React.FC = () => {

    /**
     * 组织
     */
    const [myOrgs, setMyOrgs] = React.useState<Org[]>([]);
    const [currentIdx, setCurrentIdx] = React.useState(0);

    const roleMap: Record<string, { label: string; color: string }> = {
        '1': { label: 'Operator', color: '#1890ff' },       // 蓝色
        '2': { label: 'Engineer', color: '#52c41a' },       // 绿色
        '3': { label: 'Admin', color: '#faad14' },          // 黄色
        '4': { label: 'Super Admin', color: '#f5222d' },    // 红色
    };
    const getRoleTags = (ids: string) => {
        const idTable = ids.split(',');
        return idTable.map(id => {
            const role = roleMap[id];
            return role ? (
                <span key={id} style={{
                    backgroundColor: role.color,
                    color: '#fff',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    fontSize: '12px',
                    marginRight: '6px',
                    display: 'inline-block',
                }}>
        {role.label}
      </span>
            ) : null;
        });
    }

    const getAllOrgs = () => {
        getAllMyOrgs().then(res => {
            if (res.success) {
                setMyOrgs(res.data);
                return;
            }
            err(res.message);
        }).catch(e => {
            console.error(e);
            err('something went wrong');
        });
    };

    const handleOrgChange = (event: SelectChangeEvent<string>) => {
        const selectedOrgId = event.target.value;
        const idx = myOrgs.findIndex(org => org.org_id === selectedOrgId);
        if (idx !== -1) setCurrentIdx(idx);
    };

    /**
     * 项目
     */
    const [openNewProject, setOpenNewProject] = React.useState(false);
    const [newProjectName, setNewProjectName] = React.useState("");
    const [newProjectDesc, setNewProjectDesc] = React.useState("");
    const handleOpenProject = () => {
        setOpenNewProject(true);
    }
    const handleCloseProject = () => {
        setOpenNewProject(false);
    }
    const handleCreateProject = () => {
        if (newProjectName.length < 0 || newProjectName.length > 20) {
            err('Project name must be between 1 and 20 characters');
            return
        }
        createProjectApi(Number(myOrgs[currentIdx].org_id), newProjectName, newProjectDesc).then(res => {
            if (res.success) {
                handleCloseProject();
                setNewProjectName('')
                setNewProjectDesc('')
                refreshProjects()
                return
            }
            err(res.message);
        }).catch(e=> {
            console.error(e);
            err('something went wrong');
        })
    }
    const [currentProjects, setCurrentProjects] = React.useState<Org[]>([]);
    const refreshProjects = () => {
        if (!myOrgs[currentIdx]) return;
        getAllProjectsApi(Number(myOrgs[currentIdx].org_id)).then(res=> {
            if (res.success) {
                setCurrentProjects(res.data);
                return
            }
            err(res.message);
        }).catch(e=> {
            console.error(e);
            err('something went wrong');
        })
    }
    const getFirstList = (allList: any) => {
        return allList.filter((item:any)=>{
            return Number(item.parentKey) === -1
        })
    }


    /**
     * 全局
     */
    useEffect(() => {
        getAllOrgs();
    }, []);
    useEffect(() => {
        refreshProjects();
    }, [currentIdx, myOrgs]);

    return (
        <Box className={`bg-gray-300 p-4 h-full flex w-full`}>
            <Box className={`w-full h-full bg-white shadow-md rounded-lg px-4 py-2 flex flex-col justify-between`}>
                {/*  用户加入的组织  */}
                <Box className={`w-full h-16 flex items-center justify-between`}>
                    {/* 组织名 */}
                    <Typography variant={`h6`}>
                        Current Organization: {myOrgs.length > 0 ? myOrgs[currentIdx].org_name : "Loading..."}
                    </Typography>
                    {/* 角色 */}
                    <Box className={`mx-4 h-full flex items-center justify-between`}>
                        {myOrgs.length > 0 ? getRoleTags(myOrgs[currentIdx].authorityIds) : ''}
                    </Box>
                    <Box className={`flex-1`}></Box>
                    {/* 选择组织 */}
                    <Select
                        labelId="select-label"
                        id="select"
                        value={myOrgs[currentIdx]?.org_id || ""}
                        className={`w-40`}
                        onChange={handleOrgChange}
                        size="small"
                    >
                        {myOrgs.map((org) => (
                            <MenuItem key={org.org_id} value={org.org_id}>
                                {org.org_name}
                            </MenuItem>
                        ))}
                    </Select>
                </Box>

                <Divider/>
                {/*  当前组织的项目树  */}
                <Box className={`flex-1 p-2 overflow-x-hidden overflow-y-scroll`}>
                    {
                        getFirstList(currentProjects).map((project: any, index: number) => (
                            <Accordion key={project.projectKey} className={`my-2`}>
                                <AccordionSummary
                                    expandIcon={<ExpandMoreIcon />}
                                    id={String(index)}
                                >
                                    <Box className={`flex w-full items-center justify-between`}>
                                        {/* 名称，创建者，时间，增加子项目，删除项目 */}
                                        <Typography component="h6">
                                            Project Name:<strong className={`text-blue-600 ml-2`}>{project.title}</strong>
                                        </Typography>
                                        <Box className={`flex-1`}></Box>
                                        <Typography className={`!text-[13px] !mx-2`}>Created By:
                                            <strong onClick={(e) => {
                                                e.stopPropagation();
                                                handleCopy(project.createrPhoneNum)
                                            }} className={`ml-2 text-blue-600 cursor-pointer`}>{project.createrNickName}</strong>
                                        </Typography>
                                        <Typography className={`!text-[13px] !mx-2`}>{convertToChinaTime(project.createdTime)}</Typography>
                                    </Box>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Box className={`ml-2`}>
                                        {/* Detail */}
                                        <Typography className={`!text-[13px] font-gray-500`}>{project.desc}</Typography>
                                        {/* DWGPath Analysis Upload Delete*/}
                                        <Box className={`h-12 w-full bg-amber-200 my-2`}>

                                        </Box>
                                    </Box>


                                </AccordionDetails>
                            </Accordion>
                        ))
                    }
                </Box>
                {/* 新增项目 */}
                <Divider />
                <Box className={`w-full h-16 flex items-center justify-end`}>
                    <Button variant={`contained`} onClick={handleOpenProject}>Add a Project</Button>
                </Box>
            </Box>

            {/*  弹窗  */}
            <Dialog
                open={openNewProject}
                onClose={handleCloseProject}
                aria-labelledby="alert-dialog-title"
                aria-describedby="alert-dialog-description"
            >
                <DialogTitle id="alert-dialog-title">
                    {"Create a New Project"}
                </DialogTitle>
                <DialogContent className={`flex flex-col`}>
                    <TextField onChange={(e) => setNewProjectName(e.target.value)} value={newProjectName} className={`!my-2 w-96`} id="outlined-basic" label="Project Name" size={`small`} variant="outlined" />
                    <TextField onChange={(e) => setNewProjectDesc(e.target.value)} value={newProjectDesc} className={`!my-2 w-96`} id="outlined-basic" label="Project Description" size={`small`} variant="outlined" />
                </DialogContent>
                <DialogActions>
                    <Button color={`error`} onClick={handleCloseProject} autoFocus>
                        Cancel
                    </Button>
                    <Button onClick={handleCreateProject}>Create</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default Spaces;
