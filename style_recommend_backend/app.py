from flask import Flask, request, jsonify

app = Flask(__name__)


class ColorRecommendation:
    def get_color_recommendation(self, image_path):
        # Add your image processing logic here
        # Return the color and complementary color
        color = "example_color"
        complementary_color = "example_complementary_color"
        return color, complementary_color


@app.route('/get_color_recommendation', methods=['POST'])
def get_color_recommendation():
    image_path = request.json.get('image_path')

    recommendation = ColorRecommendation().get_color_recommendation(image_path)

    return jsonify({
        'color': recommendation[0],
        'complementary_color': recommendation[1]
    })


if __name__ == '__main__':
    app.run(debug=True)
