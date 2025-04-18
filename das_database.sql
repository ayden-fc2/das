/*
 Navicat MySQL Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 90200 (9.2.0)
 Source Host           : localhost:3306
 Source Schema         : das_database

 Target Server Type    : MySQL
 Target Server Version : 90200 (9.2.0)
 File Encoding         : 65001

 Date: 18/04/2025 13:23:35
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for account_st
-- ----------------------------
DROP TABLE IF EXISTS `account_st`;
CREATE TABLE `account_st` (
  `accountId` bigint NOT NULL AUTO_INCREMENT,
  `phoneNum` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `passwordDetail` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `nickName` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  `avatar` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci DEFAULT NULL,
  PRIMARY KEY (`accountId`),
  UNIQUE KEY `phoneNum` (`phoneNum`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for block_st
-- ----------------------------
DROP TABLE IF EXISTS `block_st`;
CREATE TABLE `block_st` (
  `block_id` int NOT NULL AUTO_INCREMENT,
  `block_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `block_count` int DEFAULT NULL,
  `svg` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci,
  `handle0` int DEFAULT NULL,
  `handle1` int DEFAULT NULL,
  `dwg_id` int DEFAULT NULL,
  PRIMARY KEY (`block_id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=2253 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for insert_st
-- ----------------------------
DROP TABLE IF EXISTS `insert_st`;
CREATE TABLE `insert_st` (
  `insert_id` int NOT NULL AUTO_INCREMENT,
  `block_handle0` int DEFAULT NULL,
  `block_handle1` int DEFAULT NULL,
  `insert_handle0` int DEFAULT NULL,
  `insert_handle1` int DEFAULT NULL,
  `box_width` double DEFAULT NULL,
  `box_height` double DEFAULT NULL,
  `center_pt_x` double DEFAULT NULL,
  `center_pt_y` double DEFAULT NULL,
  `dwg_id` int DEFAULT NULL,
  `upstream` varchar(255) DEFAULT NULL,
  `downstream` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`insert_id`)
) ENGINE=InnoDB AUTO_INCREMENT=33627 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for key_pipe_st
-- ----------------------------
DROP TABLE IF EXISTS `key_pipe_st`;
CREATE TABLE `key_pipe_st` (
  `key_pipe_id` int NOT NULL AUTO_INCREMENT,
  `start_x` double DEFAULT NULL,
  `start_y` double DEFAULT NULL,
  `end_x` double DEFAULT NULL,
  `end_y` double DEFAULT NULL,
  `dwg_id` int DEFAULT NULL,
  `start_key_handle0` int DEFAULT NULL,
  `start_key_handle1` int DEFAULT NULL,
  `end_key_handle0` int DEFAULT NULL,
  `end_key_handle1` int DEFAULT NULL,
  `v_start_uuid` varchar(255) DEFAULT NULL,
  `v_end_uuid` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`key_pipe_id`)
) ENGINE=InnoDB AUTO_INCREMENT=16000 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for log_st
-- ----------------------------
DROP TABLE IF EXISTS `log_st`;
CREATE TABLE `log_st` (
  `logId` int NOT NULL AUTO_INCREMENT,
  `userId` int DEFAULT NULL,
  `copDetail` varchar(255) DEFAULT NULL,
  `copTime` datetime DEFAULT NULL,
  `copType` int DEFAULT NULL COMMENT '0表示基础操作，1表示埋点view，2表示打点',
  PRIMARY KEY (`logId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for org_st
-- ----------------------------
DROP TABLE IF EXISTS `org_st`;
CREATE TABLE `org_st` (
  `org_id` int NOT NULL AUTO_INCREMENT,
  `org_name` varchar(255) DEFAULT NULL,
  `created_time` datetime DEFAULT NULL,
  `creater_id` int DEFAULT NULL,
  `org_desc` varchar(255) DEFAULT NULL,
  `org_code` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`org_id`)
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for project_st
-- ----------------------------
DROP TABLE IF EXISTS `project_st`;
CREATE TABLE `project_st` (
  `project_key` varchar(255) DEFAULT NULL,
  `created_time` datetime DEFAULT NULL,
  `title` varchar(255) DEFAULT NULL,
  `desc` varchar(255) DEFAULT NULL,
  `parent_key` varchar(255) DEFAULT NULL,
  `children_key` varchar(255) DEFAULT NULL,
  `upload_id` int DEFAULT NULL,
  `org_id` int DEFAULT NULL,
  `creater_id` int DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ----------------------------
-- Table structure for relationship_st
-- ----------------------------
DROP TABLE IF EXISTS `relationship_st`;
CREATE TABLE `relationship_st` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `accountId` bigint NOT NULL,
  `authorityId` bigint NOT NULL COMMENT '4-超级管理员； 3-空间管理员； 2-项目管理员； 3-项目分析师',
  `orgId` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `accountId` (`accountId`),
  CONSTRAINT `relationship_st_ibfk_1` FOREIGN KEY (`accountId`) REFERENCES `account_st` (`accountId`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for upload_dwg_st
-- ----------------------------
DROP TABLE IF EXISTS `upload_dwg_st`;
CREATE TABLE `upload_dwg_st` (
  `id` int NOT NULL AUTO_INCREMENT,
  `project_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `user_id` int NOT NULL,
  `dwg_path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `json_path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL,
  `is_public` tinyint(1) NOT NULL DEFAULT '0',
  `created_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `is_deleted` int DEFAULT NULL,
  `analysised` int DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=73 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for virtual_node_st
-- ----------------------------
DROP TABLE IF EXISTS `virtual_node_st`;
CREATE TABLE `virtual_node_st` (
  `v_node_id` int NOT NULL AUTO_INCREMENT,
  `x` double DEFAULT NULL,
  `y` double DEFAULT NULL,
  `uuid` varchar(255) DEFAULT NULL,
  `dwg_id` int DEFAULT NULL,
  `finished` int DEFAULT NULL,
  PRIMARY KEY (`v_node_id`)
) ENGINE=InnoDB AUTO_INCREMENT=6842 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SET FOREIGN_KEY_CHECKS = 1;
