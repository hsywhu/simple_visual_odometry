#include "frontend/OccupancyGrid.h"
using namespace std;
using namespace cv;

OccupancyGrid::OccupancyGrid()
{
	// initializer();
	initializer1();
}

void OccupancyGrid::initializer1()
{
	OccupancyGrid::resetGrid1();
}

void OccupancyGrid::setImageSize1(size_t cols, size_t rows)
{
	Ix = cols / nx;
	Iy = rows / ny;
}

void OccupancyGrid::addPoint1(Point2f& p)
{
	int width_idx = p.x / Ix;
	int height_idx = p.y / Iy;
	isFree[width_idx][height_idx] = false;
}

bool OccupancyGrid::isNewFeature1(Point2f& p)
{
	// cout << "isNewFeature1" << Ix << " " << Iy << endl;
	int width_idx = p.x / Ix;
	int height_idx = p.y / Iy;
	
	// Center
	if (isFree[width_idx][height_idx] == false)
	{
		return false;
	}

	// Up
	if ((height_idx - 1) >= 0)
	{
		if (isFree[width_idx][height_idx - 1] == false)
		{
			return false;
		}
	}

	// Down
	if ((height_idx + 1) < ny)
	{
		if (isFree[width_idx][height_idx + 1] == false)
		{
			return false;
		}
	}

	// Left
	if ((width_idx - 1) >= 0)
	{
		if (isFree[width_idx - 1][height_idx] == false)
		{
			return false;
		}
	}

	// Right
	if ((width_idx + 1) < nx)
	{
		if (isFree[width_idx + 1][height_idx] == false)
		{
			return false;
		}
	}

	// UpRight
	if ((width_idx + 1) < nx && (height_idx - 1) >= 0)
	{
		if (isFree[width_idx + 1][height_idx - 1] == false)
		{
			return false;
		}
	}

	// UpLeft
	if ((width_idx - 1) >= 0 && (height_idx - 1) >= 0)
	{
		if (isFree[width_idx - 1][height_idx - 1] == false)
		{
			return false;
		}
	}

	// DownRight
	if ((width_idx + 1) < nx && (height_idx + 1) < ny)
	{
		if (isFree[width_idx + 1][height_idx + 1] == false)
		{
			return false;
		}
	}

	// DownLeft
	if ((width_idx - 1) >= 0 && (height_idx + 1) < ny)
	{
		if (isFree[width_idx - 1][height_idx + 1] == false)
		{
			return false;
		}
	}

	return true;
}

void OccupancyGrid::resetGrid1()
{
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			isFree[i][j] = true;
		}
	}
}

