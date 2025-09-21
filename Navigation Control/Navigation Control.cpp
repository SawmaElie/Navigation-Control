#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
#include <queue>
#include <unordered_map>
#include <algorithm>

// Global variables
cv::Point target(-1, -1); // Target point initialized to invalid (-1, -1)
int cellSize = 50; // Grid cell size in pixels

// Function to calculate the center of a marker
cv::Point2f getMarkerCenter(const std::vector<cv::Point2f>& corners) {
    float centerX = 0, centerY = 0;
    for (const auto& corner : corners) {
        centerX += corner.x;
        centerY += corner.y;
    }
    centerX /= corners.size();
    centerY /= corners.size();
    return cv::Point2f(centerX, centerY);
}

// Function to mark occupied cells with a safety margin
void markOccupiedCellsWithMargin(cv::Mat& gridImage, const std::vector<cv::Point2f>& corners, int cellSize, const cv::Scalar& color, int marginCells) {
    int minCol = INT_MAX, minRow = INT_MAX;
    int maxCol = 0, maxRow = 0;

    for (const auto& corner : corners) {
        int col = static_cast<int>(corner.x) / cellSize;
        int row = static_cast<int>(corner.y) / cellSize;

        minCol = std::min(minCol, col);
        minRow = std::min(minRow, row);
        maxCol = std::max(maxCol, col);
        maxRow = std::max(maxRow, row);
    }

    minCol = std::max(0, minCol - marginCells);
    minRow = std::max(0, minRow - marginCells);
    maxCol = std::min(gridImage.cols / cellSize - 1, maxCol + marginCells);
    maxRow = std::min(gridImage.rows / cellSize - 1, maxRow + marginCells);

    for (int r = minRow; r <= maxRow; ++r) {
        for (int c = minCol; c <= maxCol; ++c) {
            std::cout << "Marking obstacle at cell: (" << r << ", " << c << ")" << std::endl;
            cv::rectangle(gridImage,
                cv::Point(c * cellSize, r * cellSize),
                cv::Point((c + 1) * cellSize, (r + 1) * cellSize),
                color, cv::FILLED);
        }
    }
}

// Mouse callback function for selecting the target
void onMouse(int event, int x, int y, int, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Calculate the grid cell based on click coordinates
        int col = x / cellSize;
        int row = y / cellSize;

        // Update the target point
        target = cv::Point(col, row);

        std::cout << "Target selected at Grid Cell: (" << row << ", " << col << ")" << std::endl;
    }
}

// Structure to represent a node in the grid
struct Node {
    int row, col;
    float cost; // f(n) = g(n) + h(n)

    // Comparison operator for priority queue
    bool operator>(const Node& other) const {
        return cost > other.cost;
    }
};

// Function to calculate the heuristic (Euclidean distance)
float calculateHeuristic(int row, int col, int targetRow, int targetCol) {
    return std::sqrt(std::pow(row - targetRow, 2) + std::pow(col - targetCol, 2));
}

// Function to check if the robot can fit at the given position
bool canFitRobot(const cv::Mat& grid, int row, int col, int robotWidthCells, int robotHeightCells, int cellSize) {
    int rows = grid.rows / cellSize;
    int cols = grid.cols / cellSize;

    for (int r = row; r < row + robotHeightCells; ++r) {
        for (int c = col; c < col + robotWidthCells; ++c) {
            if (r >= rows || c >= cols || r < 0 || c < 0) return false; // Out of bounds

            cv::Vec3b color = grid.at<cv::Vec3b>(r * cellSize, c * cellSize);
            if (r >= rows || c >= cols || r < 0 || c < 0 || (color[0] == 0 && color[1] == 0 && color[2] == 0)) {
                return false; // Cell is an obstacle or out of bounds
            }

        }
    }
    return true; // All cells are traversable
}


// Updated A* Pathfinding algorithm
std::vector<cv::Point> aStarWithRobotSize(const cv::Mat& grid, cv::Point start, cv::Point goal, int cellSize, int robotWidthCells, int robotHeightCells) {
    int rows = grid.rows / cellSize;
    int cols = grid.cols / cellSize;

    std::vector<cv::Point> directions = {
        {0, 1}, {1, 0}, {0, -1}, {-1, 0}, // Cardinal directions
        {1, 1}, {1, -1}, {-1, 1}, {-1, -1} // Diagonal directions
    };

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openList;
    std::unordered_map<int, bool> closedSet;
    std::unordered_map<int, float> gScore;
    std::unordered_map<int, cv::Point> cameFrom;

    auto toKey = [cols](int row, int col) { return row * cols + col; };

    gScore[toKey(start.y, start.x)] = 0;
    openList.push({ start.y, start.x, calculateHeuristic(start.y, start.x, goal.y, goal.x) });

    while (!openList.empty()) {
        Node current = openList.top();
        openList.pop();

        int currentKey = toKey(current.row, current.col);

        if (current.row == goal.y && current.col == goal.x) {
            std::vector<cv::Point> path;
            cv::Point currentNode = goal;

            while (cameFrom.find(toKey(currentNode.y, currentNode.x)) != cameFrom.end()) {
                path.push_back(currentNode);
                currentNode = cameFrom[toKey(currentNode.y, currentNode.x)];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        closedSet[currentKey] = true;

        for (const cv::Point& dir : directions) {
            int neighborRow = current.row + dir.y;
            int neighborCol = current.col + dir.x;

            if (neighborRow < 0 || neighborRow >= rows || neighborCol < 0 || neighborCol >= cols) {
                continue;
            }

            if (!canFitRobot(grid, neighborRow, neighborCol, robotWidthCells, robotHeightCells, cellSize)) {
                continue; // Skip if the robot cannot fit in the neighbor
            }

            int neighborKey = toKey(neighborRow, neighborCol);

            if (closedSet.find(neighborKey) != closedSet.end()) {
                continue;
            }

            float tentativeG = gScore[currentKey] + calculateHeuristic(current.row, current.col, neighborRow, neighborCol);

            if (gScore.find(neighborKey) == gScore.end() || tentativeG < gScore[neighborKey]) {
                gScore[neighborKey] = tentativeG;
                float fScore = tentativeG + calculateHeuristic(neighborRow, neighborCol, goal.y, goal.x);
                openList.push({ neighborRow, neighborCol, fScore });
                cameFrom[neighborKey] = cv::Point(current.col, current.row);
            }
        }
    }

    return {}; // Return empty path if no solution
}



int main() {
    cv::Mat image = cv::imread("C:/Users/M.M/Downloads/map.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not load the image!" << std::endl;
        return -1;
    }

    // Known marker sizes in meters
    const float obstacleMarkerSize = 0.175; // 17.5 cm
    const float robotMarkerSize = 0.177;    // 17.7 cm

    // Initialize ArUco dictionaries and detectors
    cv::aruco::Dictionary dictionary1 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50); // Obstacles
    cv::aruco::Dictionary dictionary2 = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250); // Robot

    cv::aruco::DetectorParameters parameters;
    cv::aruco::ArucoDetector detector1(dictionary1, parameters); // Detector for obstacles
    cv::aruco::ArucoDetector detector2(dictionary2, parameters); // Detector for the robot

    // Variables for marker detection
    std::vector<std::vector<cv::Point2f>> markerCorners1, markerCorners2;
    std::vector<int> markerIds1, markerIds2;

    detector1.detectMarkers(image, markerCorners1, markerIds1);
    detector2.detectMarkers(image, markerCorners2, markerIds2);

    // Calculate scale factors (pixels per meter)
    float obstacleScale = 1.0;
    if (!markerCorners1.empty()) {
        float pixelWidth = cv::norm(markerCorners1[0][0] - markerCorners1[0][1]); // Width in pixels
        obstacleScale = obstacleMarkerSize / pixelWidth;
    }

    cv::Point robotStart(-1, -1); // Default start position
    std::vector<cv::Point> robotOccupiedCells; // Detected cells for the robot

    if (!markerCorners2.empty()) {
        // Calculate the robot's detected cells
        int minCol = INT_MAX, minRow = INT_MAX;
        int maxCol = 0, maxRow = 0;

        for (const auto& corner : markerCorners2[0]) {
            int col = static_cast<int>(corner.x) / cellSize;
            int row = static_cast<int>(corner.y) / cellSize;
            minCol = std::min(minCol, col);
            minRow = std::min(minRow, row);
            maxCol = std::max(maxCol, col);
            maxRow = std::max(maxRow, row);
        }

        for (int r = minRow; r <= maxRow; ++r) {
            for (int c = minCol; c <= maxCol; ++c) {
                robotOccupiedCells.push_back(cv::Point(c, r));

            }
        }
        // Output the occupied cells by the robot
        std::cout << "Robot occupied cells:" << std::endl;
        for (const auto& cell : robotOccupiedCells) {
            std::cout << "(" << cell.y << ", " << cell.x << ")" << std::endl;
        }

        // Set the robot's starting point as the center of the detected cells
        robotStart = cv::Point((minCol + maxCol) / 2, (minRow + maxRow) / 2);
        std::cout << "Robot detected at Grid Cell: (" << robotStart.y << ", " << robotStart.x << ")" << std::endl;
        // Convert the grid cell to real-world coordinates
        float realX = (robotStart.x * cellSize + cellSize / 2.0) * obstacleScale;
        float realY = (robotStart.y * cellSize + cellSize / 2.0) * obstacleScale;
        std::cout << "Robot detected at Real-World Coordinates: (" << realY << " meters, " << realX << " meters)" << std::endl;
    }
    else {
        std::cerr << "Robot marker not detected! Please ensure the ArUco marker is visible." << std::endl;
        return -1; // Exit if the robot's position cannot be detected
    }

    // Initialize occupancy grid
    int gridRows = (image.rows + cellSize - 1) / cellSize;
    int gridCols = (image.cols + cellSize - 1) / cellSize;

    cv::Mat gridImage(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the grid
    for (int y = 0; y <= image.rows; y += cellSize) {
        cv::line(gridImage, cv::Point(0, y), cv::Point(image.cols - 1, y), cv::Scalar(0, 255, 0), 1);
    }
    for (int x = 0; x <= image.cols; x += cellSize) {
        cv::line(gridImage, cv::Point(x, 0), cv::Point(x, image.rows - 1), cv::Scalar(0, 255, 0), 1);
    }

    // Mark the first 7 cells on the left as black (obstacles)
    for (int y = 0; y < gridImage.rows; y += cellSize) {
        for (int x = 0; x < 7 * cellSize; x += cellSize) {
            cv::rectangle(gridImage,
                cv::Point(x, y),
                cv::Point(x + cellSize, y + cellSize),
                cv::Scalar(0, 0, 0), cv::FILLED);
        }
    }

    // Mark the last 9 cells on the right as black (obstacles)
    for (int y = 0; y < gridImage.rows; y += cellSize) {
        for (int x = gridImage.cols - 9 * cellSize; x < gridImage.cols; x += cellSize) {
            cv::rectangle(gridImage,
                cv::Point(x, y),
                cv::Point(x + cellSize, y + cellSize),
                cv::Scalar(0, 0, 0), cv::FILLED);
        }
    }

    const int safetyMarginCells = 2;

    // Mark obstacles on the grid
    for (size_t i = 0; i < markerCorners1.size(); ++i) {
        markOccupiedCellsWithMargin(gridImage, markerCorners1[i], cellSize, cv::Scalar(0, 0, 0), safetyMarginCells);
    }

    // Mark the robot's initial position on the grid
    for (const auto& cell : robotOccupiedCells) {
        cv::rectangle(gridImage,
            cv::Point(cell.x * cellSize, cell.y * cellSize),
            cv::Point((cell.x + 1) * cellSize, (cell.y + 1) * cellSize),
            cv::Scalar(255, 0, 0), cv::FILLED);
    }

    // Set mouse callback for manual target selection
    cv::namedWindow("Occupancy Grid", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Occupancy Grid", onMouse);

    while (true) {
        cv::Mat displayGrid = gridImage.clone();

        if (target.x != -1 && target.y != -1) {
            // Mark the selected target on the grid
            cv::rectangle(displayGrid,
                cv::Point(target.x * cellSize, target.y * cellSize),
                cv::Point((target.x + 1) * cellSize, (target.y + 1) * cellSize),
                cv::Scalar(0, 255, 0), cv::FILLED);

            // Perform A* Pathfinding immediately after the target is selected
            if (robotStart.x != -1) {
                int robotWidthCells = (robotOccupiedCells.back().x - robotOccupiedCells.front().x);
                int robotHeightCells = (robotOccupiedCells.back().y - robotOccupiedCells.front().y);

                std::vector<cv::Point> path = aStarWithRobotSize(gridImage, robotStart, target, cellSize, robotWidthCells, robotHeightCells);
                if (!path.empty()) {
                    std::cout << "Path coordinates :" << std::endl;
                    for (const cv::Point& p : path) {
                        //std::cout << "(" << p.y << ", " << p.x << ")" << std::endl; // Print grid coordinates
                        // Convert to real-world coordinates
                        float realX = (p.x * cellSize + cellSize / 2.0) * obstacleScale; // X-coordinate in meters
                        float realY = (p.y * cellSize + cellSize / 2.0) * obstacleScale; // Y-coordinate in meters
                        std::cout << "{" << realY << " , " << realX << "}," << std::endl;
                        cv::rectangle(displayGrid,
                            cv::Point(p.x * cellSize, p.y * cellSize),
                            cv::Point((p.x + 1) * cellSize, (p.y + 1) * cellSize),
                            cv::Scalar(0, 0, 255), cv::FILLED);

                    }
                    for (const cv::Point& p : path) {
                        // Move all robot cells along the path
                        for (const auto& cell : robotOccupiedCells) {
                            int dx = cell.x - robotStart.x;
                            int dy = cell.y - robotStart.y;

                            cv::Point movedCell(p.x + dx, p.y + dy);
                            cv::rectangle(displayGrid,
                                cv::Point(movedCell.x * cellSize, movedCell.y * cellSize),
                                cv::Point((movedCell.x + 1) * cellSize, (movedCell.y + 1) * cellSize),
                                cv::Scalar(0, 0, 255), cv::FILLED);
                        }
                    }

                    // Display the updated grid with the path in a new window
                    cv::namedWindow("Path Planning", cv::WINDOW_NORMAL);
                    cv::imshow("Path Planning", displayGrid);
                    cv::waitKey(0);
                    cv::destroyWindow("Path Planning");
                }
                else {
                    std::cout << "No path found!" << std::endl;
                }

                break; // Exit the loop after showing the path
            }
            else {
                std::cerr << "Robot start position is invalid." << std::endl;
                break;
            }
        }

        // Display the occupancy grid
        cv::imshow("Occupancy Grid", displayGrid);

        // Allow user to exit the loop by pressing 'q'
        char key = cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    cv::destroyAllWindows(); // Ensure all windows are closed at the end
    return 0;
}
