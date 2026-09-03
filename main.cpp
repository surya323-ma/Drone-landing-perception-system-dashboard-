// dev/creator=tubakhxn
// Build: g++ -std=c++17 hud_single.cpp -o hud_single `pkg-config --cflags --libs opencv4`
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <cmath>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace hud {

struct Palette {
    static constexpr cv::Scalar CYAN      {255, 240, 60};
    static constexpr cv::Scalar CYAN_SOFT {230, 200, 90};
    static constexpr cv::Scalar TEAL      {200, 180, 40};
    static constexpr cv::Scalar GREEN     {140, 235, 120};
    static constexpr cv::Scalar WHITE     {245, 245, 245};
    static constexpr cv::Scalar WHITE_DIM {190, 190, 190};
    static constexpr cv::Scalar AMBER     {60, 170, 235};
    static constexpr cv::Scalar BG_PANEL  {28, 22, 18};
    static constexpr cv::Scalar GRID      {90, 70, 50};
};

inline double clampd(double v, double a, double b) { return std::max(a, std::min(b, v)); }

inline void dashed_line(cv::Mat& img, cv::Point2d p1, cv::Point2d p2,
                         const cv::Scalar& color, int thickness = 2,
                         double dash_len = 10.0, double gap_len = 8.0, double phase = 0.0) {
    cv::Point2d diff = p2 - p1;
    double dist = std::hypot(diff.x, diff.y);
    if (dist < 1.0) return;
    cv::Point2d dir(diff.x / dist, diff.y / dist);
    double step = dash_len + gap_len;
    double d = -std::fmod(phase, step);
    while (d < dist) {
        double s = std::max(d, 0.0), e = std::min(d + dash_len, dist);
        if (e > s) {
            cv::Point2d a = p1 + dir * s, b = p1 + dir * e;
            cv::line(img, cv::Point((int)a.x, (int)a.y), cv::Point((int)b.x, (int)b.y), color, thickness, cv::LINE_AA);
        }
        d += step;
    }
}

inline void glow_circle(cv::Mat& img, cv::Point2d center, double radius,
                         const cv::Scalar& color, double intensity = 1.0) {
    cv::Mat overlay = img.clone();
    for (auto& l : std::vector<std::pair<double,double>>{{radius*2.2,0.10},{radius*1.5,0.18},{radius,0.35}})
        cv::circle(overlay, cv::Point((int)center.x,(int)center.y), (int)l.first, color, -1, cv::LINE_AA);
    cv::addWeighted(overlay, 0.5*intensity, img, 1-0.5*intensity, 0, img);
    cv::circle(img, cv::Point((int)center.x,(int)center.y), std::max(2,(int)(radius*0.55)), color, -1, cv::LINE_AA);
}

inline void rounded_rect(cv::Mat& img, cv::Point pt1, cv::Point pt2,
                          const cv::Scalar& color, int thickness = -1, int radius = 10, double alpha = 1.0) {
    cv::Mat overlay = img.clone();
    if (thickness < 0) {
        cv::rectangle(overlay, {pt1.x+radius,pt1.y}, {pt2.x-radius,pt2.y}, color, -1, cv::LINE_AA);
        cv::rectangle(overlay, {pt1.x,pt1.y+radius}, {pt2.x,pt2.y-radius}, color, -1, cv::LINE_AA);
        for (auto& c : std::vector<cv::Point>{{pt1.x+radius,pt1.y+radius},{pt2.x-radius,pt1.y+radius},
                                               {pt1.x+radius,pt2.y-radius},{pt2.x-radius,pt2.y-radius}})
            cv::circle(overlay, c, radius, color, -1, cv::LINE_AA);
    } else {
        cv::rectangle(overlay, pt1, pt2, color, thickness, cv::LINE_AA);
    }
    cv::addWeighted(overlay, alpha, img, 1-alpha, 0, img);
}

inline void put_text(cv::Mat& img, const std::string& text, cv::Point org,
                      double scale = 0.5, const cv::Scalar& color = Palette::WHITE,
                      int thickness = 1, bool glow = false) {
    if (glow) cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0,0,0), thickness+2, cv::LINE_AA);
    cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

inline void frame_corners(cv::Mat& img, int margin = 24, int len = 26,
                           const cv::Scalar& color = Palette::CYAN_SOFT, int thickness = 2) {
    int w = img.cols, h = img.rows;
    for (auto& c : std::vector<std::array<int,4>>{{margin,margin,1,1},{w-margin,margin,-1,1},
                                                   {margin,h-margin,1,-1},{w-margin,h-margin,-1,-1}}) {
        cv::line(img, {c[0],c[1]}, {c[0]+c[2]*len,c[1]}, color, thickness, cv::LINE_AA);
        cv::line(img, {c[0],c[1]}, {c[0],c[1]+c[3]*len}, color, thickness, cv::LINE_AA);
    }
}

inline void scanline(cv::Mat& img, double t_sec, double speed = 0.35,
                      const cv::Scalar& color = Palette::CYAN, double alpha = 0.08) {
    int h = img.rows, w = img.cols;
    int y = (int)((std::sin(t_sec*speed)*0.5+0.5)*h);
    cv::Mat overlay = img.clone();
    cv::line(overlay, {0,y}, {w,y}, color, 1, cv::LINE_AA);
    cv::addWeighted(overlay, alpha, img, 1-alpha, 0, img);
}

class TrailRenderer {
public:
    explicit TrailRenderer(size_t max_len = 45) : max_len_(max_len) {}
    void push(cv::Point2d pt) { pts_.push_back(pt); while (pts_.size() > max_len_) pts_.pop_front(); }
    void draw(cv::Mat& img, const cv::Scalar& color = Palette::CYAN_SOFT) const {
        size_t n = pts_.size();
        if (n < 2) return;
        for (size_t i = 1; i < n; i++) {
            double t = (double)i/n, a = 0.15+0.55*t;
            cv::Scalar c(color[0]*a, color[1]*a, color[2]*a);
            cv::line(img, cv::Point((int)pts_[i-1].x,(int)pts_[i-1].y), cv::Point((int)pts_[i].x,(int)pts_[i].y), c, 1, cv::LINE_AA);
        }
    }
private:
    size_t max_len_;
    std::deque<cv::Point2d> pts_;
};

} // namespace hud

int main() {
    const int W = 800, H = 500, FPS = 30;
    const int N = (int)(8.0 * FPS);

    cv::VideoWriter writer("kalman_ball_output.mp4", cv::VideoWriter::fourcc('m','p','4','v'), FPS, cv::Size(W,H));

    double bx = 100, by = 100, vx = 220, vy = 0;
    const double gravity = 480.0, radius = 14.0, dt = 1.0/FPS;

    cv::KalmanFilter kf(4, 2);
    kf.transitionMatrix = (cv::Mat_<float>(4,4) << 1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1);
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(4.0));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(6.0));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));
    kf.statePost = (cv::Mat_<float>(4,1) << (float)bx,(float)by,(float)vx,(float)vy);

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 5.0);

    hud::TrailRenderer measured_trail(60), filtered_trail(60);

    for (int i = 0; i < N; i++) {
        double t_sec = i/(double)FPS;

        vy += gravity*dt;
        bx += vx*dt;
        by += vy*dt;
        if (by > H-radius) { by = H-radius; vy *= -0.72; }
        if (bx < radius || bx > W-radius) { vx *= -1; bx = hud::clampd(bx, radius, W-radius); }

        double mx = bx + noise(rng), my = by + noise(rng);

        kf.predict();
        cv::Mat measurement = (cv::Mat_<float>(2,1) << (float)mx,(float)my);
        cv::Mat estimate = kf.correct(measurement);

        cv::Point2d measured_pt(mx,my), filtered_pt(estimate.at<float>(0), estimate.at<float>(1));
        measured_trail.push(measured_pt);
        filtered_trail.push(filtered_pt);

        cv::Mat frame(H, W, CV_8UC3, cv::Scalar(20,16,14));
        measured_trail.draw(frame, hud::Palette::WHITE_DIM);
        filtered_trail.draw(frame, hud::Palette::CYAN);
        cv::circle(frame, cv::Point((int)mx,(int)my), 3, hud::Palette::AMBER, -1, cv::LINE_AA);
        hud::glow_circle(frame, filtered_pt, radius, hud::Palette::CYAN, 0.7);

        hud::rounded_rect(frame, {16,16}, {330,78}, hud::Palette::BG_PANEL, -1, 8, 0.5);
        hud::put_text(frame, "KALMAN FILTER: BOUNCING BALL", {28,38}, 0.5, hud::Palette::WHITE, 1, true);
        hud::put_text(frame, "amber = noisy measurement, cyan = filtered estimate", {28,60}, 0.38, hud::Palette::WHITE_DIM, 1, true);

        hud::frame_corners(frame);
        hud::scanline(frame, t_sec);
        writer.write(frame);
    }

    writer.release();
    std::cout << "Wrote kalman_ball_output.mp4\n";
    return 0;
}