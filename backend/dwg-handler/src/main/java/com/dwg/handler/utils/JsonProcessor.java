package com.dwg.handler.utils;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

@Component
public class JsonProcessor {
    /**
     * 读取 JSON 文件并修复编码问题
     */
    public JSONObject readJsonFile(String filePath) {
        try {
            String jsonContent = new String(Files.readAllBytes(Paths.get(filePath)));
            return JSON.parseObject(jsonContent);
        } catch (IOException e) {
            throw new RuntimeException("无法读取 JSON 文件: " + e.getMessage(), e);
        }
    }

    /**
     * 根据句柄和jsonData来获取对应Object的公共方法
     */
    public JSONObject findBlockHeaderByHandle(JSONObject jsonData, List<Integer> targetHandle) {
        JSONArray objects = jsonData.getJSONArray("OBJECTS");
        for (int i = 0; i < objects.size(); i++) {
            JSONObject obj = objects.getJSONObject(i);
            JSONArray handle = obj.getJSONArray("handle");
            if (handle.size() != 3) continue;

            List<Integer> currentHandle = Arrays.asList(
                    handle.getInteger(0),
                    handle.getInteger(1),
                    handle.getInteger(2)
            );

            if (currentHandle.equals(targetHandle)) {
                return obj;
            }
        }
        return null;
    }

    /**
     * 获取 BLOCK_CONTROL
     */
    public JSONObject getBlockControl(JSONObject jsonData) {
        JSONArray objects = jsonData.getJSONArray("OBJECTS");
        JSONObject blockControl = null;
        for (int i = 0; i < objects.size(); i++) {
            JSONObject obj = objects.getJSONObject(i);
            if ("BLOCK_CONTROL".equals(obj.getString("object"))) {
                blockControl = obj;
                return blockControl;
            }
        }
        return null;
    }

    /**
     * 获取所有 BLOCK_HEADER
     */
    public JSONArray collectBlockHeaders(JSONObject jsonData) {
        JSONArray result = new JSONArray();

        // 1. 找到 BLOCK_CONTROL 对象
        JSONObject blockControl = getBlockControl(jsonData);
        if (blockControl == null) return result;

        // 2. 提取 entries 并生成目标句柄列表
        JSONArray entries = blockControl.getJSONArray("entries");
        Set<List<Integer>> targetHandles = new HashSet<>();
        for (int i = 0; i < entries.size(); i++) {
            JSONArray entry = entries.getJSONArray(i);
            if (entry.size() < 4) continue; // 无效格式

            int num1 = entry.getInteger(1);
            int num2 = entry.getInteger(2);
            if (num1 == 0 && num2 == 0) continue; // 过滤无效句柄

            List<Integer> newHandle = Arrays.asList(0, num1, num2);
            targetHandles.add(newHandle);
        }

        // 3. 匹配 BLOCK_HEADER 的 handle
        JSONArray objects = jsonData.getJSONArray("OBJECTS");
        for (int i = 0; i < objects.size(); i++) {
            JSONObject obj = objects.getJSONObject(i);
            if (!"BLOCK_HEADER".equals(obj.getString("object"))) continue;

            JSONArray handle = obj.getJSONArray("handle");
            if (handle.size() != 3) continue; // 确保是三维句柄

            List<Integer> handleList = Arrays.asList(
                    handle.getInteger(0),
                    handle.getInteger(1),
                    handle.getInteger(2)
            );

            if (targetHandles.contains(handleList)) {
                result.add(obj);
            }
        }

        return result;
    }

    /**
     * 获取 model_space 的 BLOCK_HEADER
     */
    public JSONObject getModelSpaceBlock(JSONObject jsonData) {
        JSONObject blockControl = getBlockControl(jsonData);
        if (blockControl == null) return null;

        // 1. 获取 model_space 句柄
        JSONArray modelSpaceHandleArray = blockControl.getJSONArray("model_space");
        List<Integer> modelSpaceHandle = Arrays.asList(
                0, // 通常 model_space 的 handle 第一部分为 0
                modelSpaceHandleArray.getInteger(1),
                modelSpaceHandleArray.getInteger(2)
        );

        // 2. 查找匹配的 BLOCK_HEADER
        return findBlockHeaderByHandle(jsonData, modelSpaceHandle);
    }

    /**
     * 收集指定 BLOCK_HEADER 内的所有实体
     */
    public JSONArray collectEntitiesInBlock(JSONObject jsonData, JSONObject blockHeader) {
        JSONArray entities = new JSONArray();
        if (!blockHeader.containsKey("entities")) return entities;

        // 1. 获取 entities 句柄列表
        JSONArray entityHandles = blockHeader.getJSONArray("entities");
        JSONArray allObjects = jsonData.getJSONArray("OBJECTS");

        // 2. 遍历所有 OBJECTS 查找匹配实体
        for (int i = 0; i < entityHandles.size(); i++) {
            JSONArray handleArray = entityHandles.getJSONArray(i);
            List<Integer> targetHandle = Arrays.asList(
                    0,
                    handleArray.getInteger(1),
                    handleArray.getInteger(2)
            );

            for (int j = 0; j < allObjects.size(); j++) {
                JSONObject obj = allObjects.getJSONObject(j);
                if (!obj.containsKey("handle")) continue;

                JSONArray objHandleArray = obj.getJSONArray("handle");
                if (objHandleArray.size() < 3) continue;

                List<Integer> currentHandle = Arrays.asList(
                        objHandleArray.getInteger(0),
                        objHandleArray.getInteger(1),
                        objHandleArray.getInteger(2)
                );

                if (currentHandle.equals(targetHandle)) {
                    entities.add(obj);
                    break;
                }
            }
        }
        return entities;
    }

    // 新增方法：查找所有INSERT实体
    private JSONArray findAllInserts(JSONObject jsonData) {
        JSONArray inserts = new JSONArray();
        JSONArray objects = jsonData.getJSONArray("OBJECTS");
        for (int i = 0; i < objects.size(); i++) {
            JSONObject obj = objects.getJSONObject(i);
            if ("INSERT".equals(obj.getString("entity"))) {
                inserts.add(obj);
            }
        }
        return inserts;
    }

    // 新增方法：检查块是否被INSERT引用
    private boolean hasInsertReference(JSONArray inserts, List<Integer> targetHandle) {
        for (Object obj : inserts) {
            JSONObject insert = (JSONObject) obj;
            JSONArray blockHeader = insert.getJSONArray("block_header");
            List<Integer> refHandle = Arrays.asList(
                    0,
                    blockHeader.getInteger(1),
                    blockHeader.getInteger(2)
            );
            if (refHandle.equals(targetHandle)) {
                return true;
            }
        }
        return false;
    }

    private JSONArray findInsertsForBlock(JSONArray allInserts, List<Integer> targetBlockHandle) {
        JSONArray result = new JSONArray();

        for (Object obj : allInserts) {
            JSONObject insert = (JSONObject) obj;

            // 1. 获取 INSERT 引用的 block_header 句柄
            JSONArray blockHeader = insert.getJSONArray("block_header");
            if (blockHeader.size() < 4) continue; // 无效格式

            // 2. 转换为三维句柄格式 [0, x, y]
            List<Integer> refHandle = Arrays.asList(
                    0, // 固定补零
                    blockHeader.getInteger(1), // 取第2位
                    blockHeader.getInteger(2)  // 取第3位
            );

            // 3. 匹配目标句柄
            if (refHandle.equals(targetBlockHandle)) {
                result.add(insert);
            }
        }
        return result;
    }


    // 新增方法：处理INSERT实例的坐标变换
//    private JSONArray processInserts(JSONObject jsonData, JSONArray inserts, JSONArray originalEntities) {
//        JSONArray result = new JSONArray();
//
//        for (Object obj : inserts) {
//            JSONObject insert = (JSONObject) obj;
//            JSONObject instance = new JSONObject();
//
//            // 提取变换参数
//            double[] insPt = parsePoint(insert.getJSONArray("ins_pt"));
//            double[] scale = parseScale(insert.getJSONArray("scale"));
//            double rotation = insert.getDoubleValue("rotation");
//
//            // 应用变换到原始实体
//            JSONArray transformed = new JSONArray();
//            for (Object entityObj : originalEntities) {
//                JSONObject entity = (JSONObject) ((JSONObject) entityObj).clone();
//
//                // 点坐标变换（示例处理LWPOLYLINE）
//                if ("LWPOLYLINE".equals(entity.getString("entity"))) {
//                    JSONArray points = entity.getJSONArray("points");
//                    JSONArray transformedPoints = new JSONArray();
//
//                    for (Object pointObj : points) {
//                        JSONArray point = (JSONArray) pointObj;
//                        double x = point.getDoubleValue(0) * scale[0];
//                        double y = point.getDoubleValue(1) * scale[1];
//
//                        // 旋转变换（绕原点）
//                        double cos = Math.cos(rotation);
//                        double sin = Math.sin(rotation);
//                        double xRot = x * cos - y * sin;
//                        double yRot = x * sin + y * cos;
//
//                        // 平移变换
//                        JSONArray newPoint = new JSONArray()
//                                .add(xRot + insPt[0])
//                                .add(yRot + insPt[1]);
//
//                        transformedPoints.add(newPoint);
//                    }
//                    entity.put("points", transformedPoints);
//                }
//                transformed.add(entity);
//            }
//
//            instance.put("insert_params", insert);
//            instance.put("transformed_entities", transformed);
//            result.add(instance);
//        }
//        return result;
//    }

    // 坐标解析工具方法
    private double[] parsePoint(JSONArray arr) {
        return new double[]{
                arr.getDoubleValue(0),
                arr.getDoubleValue(1),
                arr.size() > 2 ? arr.getDoubleValue(2) : 0
        };
    }

    private double[] parseScale(JSONArray arr) {
        return new double[]{
                arr.getDoubleValue(0),
                arr.getDoubleValue(1),
                arr.size() > 2 ? arr.getDoubleValue(2) : 1
        };
    }

    /**
     * 保存 JSON 文件
     */
    public void saveJsonToFile(JSONObject jsonData, String filePath) {
        try {
            Files.write(Paths.get(filePath), JSON.toJSONString(jsonData, true).getBytes());
            System.out.println("JSON 处理完成，已保存到: " + filePath);
        } catch (IOException e) {
            throw new RuntimeException("无法写入 JSON 文件: " + e.getMessage(), e);
        }
    }

    /**
     * 简化实体的属性，只保留必要的属性
     */
    public JSONObject simplifyEntity(JSONObject entity) {
        if (entity.containsKey("$ref")) {
            System.out.println(entity.getString("$ref"));
            return entity;
        }
        JSONObject result = new JSONObject();
        result.put("entity", entity.getString("entity"));
        copyIfExists(entity, result, "handle");
        copyIfExists(entity, result, "ins_pt");
        copyIfExists(entity, result, "layer");
        copyIfExists(entity, result, "color");
        copyIfExists(entity, result, "style");
        copyIfExists(entity, result, "text_value");
        copyIfExists(entity, result, "center");
        copyIfExists(entity, result, "start_angle");
        copyIfExists(entity, result, "end_angle");
        copyIfExists(entity, result, "radius");
        copyIfExists(entity, result, "block_header");
        copyIfExists(entity, result, "rotation");
        copyIfExists(entity, result, "scale");
        copyIfExists(entity, result, "start");
        copyIfExists(entity, result, "end");
        copyIfExists(entity, result, "points");
        copyIfExists(entity, result, "flag");
        copyIfExists(entity, result, "text");
        return result;

    }

    private void copyIfExists(JSONObject source, JSONObject target, String key) {
        if (source.containsKey(key)) {
            Object value = source.get(key);  // 保留原始类型
            target.put(key, value);
        }
    }

    /**
     * 运行所有处理步骤
     */
    public void processJsonFile(String inputFile, String outputFile) {
        JSONObject jsonData = readJsonFile(inputFile);
        JSONObject resultData = new JSONObject();

        // 获取到所有的blockControl和所有blockHeaders
        JSONObject blockControl = getBlockControl(jsonData);
        JSONArray blockHeaders = collectBlockHeaders(jsonData);
//        System.out.println("Block Control: " + blockControl);
//        System.out.println("All Block Headers: " + blockHeaders);

        // 渲染有效的blockHeaders
        JSONArray usedBlocks = new JSONArray();
        JSONArray allInserts = findAllInserts(jsonData);
        for (Object obj : blockHeaders) {
            JSONObject block = (JSONObject) obj;
            String blockName = block.getString("name");
            List<Integer> blockHandle = Arrays.asList(
                    0,
                    block.getJSONArray("handle").getInteger(1),
                    block.getJSONArray("handle").getInteger(2)
            );

            // 检查是否有INSERT引用此块
            if (hasInsertReference(allInserts, blockHandle)) {
                JSONObject blockData = new JSONObject();
                blockData.put("name", blockName);
                blockData.put("base_point", block.getJSONArray("base_pt"));

                // 收集块内原始实体（未变换）
                JSONArray originalEntities = collectEntitiesInBlock(jsonData, block);
                blockData.put("original_entities", originalEntities);

                // 收集所有引用此块的INSERT实例
                JSONArray inserts = findInsertsForBlock(allInserts, blockHandle);
//                blockData.put("inserts", processInserts(jsonData, inserts, originalEntities));
                blockData.put("inserts", inserts);
                blockData.put("handle", blockHandle);

                usedBlocks.add(blockData);
            }
        }
        resultData.put("USED_BLOCKS", usedBlocks);


        // 渲染model_space
//        System.out.println("开始处理model_space!");
        JSONObject modelSpaceBlock = getModelSpaceBlock(jsonData);
//        System.out.println("获取到model_space!" + modelSpaceBlock);
        if (modelSpaceBlock != null) {
            JSONArray modelSpaceEntities = collectEntitiesInBlock(jsonData, modelSpaceBlock);
            resultData.put("MODEL_SPACE", modelSpaceEntities);
            // System.out.println("model_space_entities: " + modelSpaceEntities);
        }
//        System.out.println("model_space处理完成！");
        
        
        // 保存文件
        saveJsonToFile(resultData, outputFile);
//        saveJsonToFile(resultData, "tmp.json");
//        JSONObject jsonDataToSim = readJsonFile("tmp.json");
//        // 统一简化JSON数据
//        for (int i = 0; i < jsonDataToSim.getJSONArray("MODEL_SPACE").size(); i++) {
//            JSONObject entity = jsonDataToSim.getJSONArray("MODEL_SPACE").getJSONObject(i);
//            jsonDataToSim.getJSONArray("MODEL_SPACE").set(i, simplifyEntity(entity));
//        }
//        for (int i = 0; i < jsonDataToSim.getJSONArray("USED_BLOCKS").size(); i++) {
//            for (int j = 0; j < jsonDataToSim.getJSONArray("USED_BLOCKS").getJSONObject(i).getJSONArray("original_entities").size(); j++) {
//                JSONObject entity = jsonDataToSim.getJSONArray("USED_BLOCKS").getJSONObject(i).getJSONArray("original_entities").getJSONObject(j);
//                jsonDataToSim.getJSONArray("USED_BLOCKS").getJSONObject(i).getJSONArray("original_entities").set(j, simplifyEntity(entity));
//            }
//            for (int j = 0; j < jsonDataToSim.getJSONArray("USED_BLOCKS").getJSONObject(i).getJSONArray("inserts").size(); j++) {
//                JSONObject insert = jsonDataToSim.getJSONArray("USED_BLOCKS").getJSONObject(i).getJSONArray("inserts").getJSONObject(j);
//                jsonDataToSim.getJSONArray("USED_BLOCKS").getJSONObject(i).getJSONArray("inserts").set(j, simplifyEntity(insert));
//            }
//        }
//        saveJsonToFile(jsonDataToSim, outputFile);
    }
}