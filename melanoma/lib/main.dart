// @dart=2.9
import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:flutter_speed_dial/flutter_speed_dial.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File _image;
  File croppedFile;
  final picker = ImagePicker();
  var serverReceiverPath = "https://skin-apps.herokuapp.com/upload-api";
  String result2;
  String test = '';
  String test2;
  String bcc2;
  String kulitNormal2;
  String melanoma2;

  var score;

  Future getImage() async {
    final pickedFile = await picker.getImage(
        source: ImageSource.gallery,
        imageQuality: 100,
        maxHeight: 1024,
        maxWidth: 1024);

    if (pickedFile != null) {
      croppedFile = await ImageCropper.cropImage(
          sourcePath: pickedFile.path,
          aspectRatioPresets: [
            CropAspectRatioPreset.square,
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio3x2,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
          androidUiSettings: AndroidUiSettings(
            toolbarTitle: 'Crop',
            toolbarColor: Colors.blue,
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: false,
          ));
    } else {
      print('No image selected.');
    }

    setState(() {
      if (croppedFile != null) {
        _image = croppedFile;
        uploadImage(croppedFile.path);
        test2 = 'loading...';
        score = '';
        bcc2 = '';
        kulitNormal2 = '';
        melanoma2 = '';
      } else {
        print('Cancel Image');
      }
    });
  }

  Future getCamera() async {
    final pickedFile = await picker.getImage(
        source: ImageSource.camera,
        imageQuality: 100,
        maxHeight: 1024,
        maxWidth: 1024);

    if (pickedFile != null) {
      croppedFile = await ImageCropper.cropImage(
          sourcePath: pickedFile.path,
          aspectRatioPresets: [
            CropAspectRatioPreset.square,
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio3x2,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
          androidUiSettings: AndroidUiSettings(
            toolbarTitle: 'Crop',
            toolbarColor: Colors.blue,
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: false,
          ));
    } else {
      print('No image selected.');
    }

    setState(() {
      if (croppedFile != null) {
        _image = croppedFile;
        uploadImage(croppedFile.path);
        test2 = 'loading...';
        score = '';
        bcc2 = '';
        kulitNormal2 = '';
        melanoma2 = '';
      } else {
        print('Cancel Image');
      }
    });
  }

  Future<Result> uploadImage(filename) async {
    var request = http.MultipartRequest('POST', Uri.parse(serverReceiverPath));
    request.files.add(await http.MultipartFile.fromPath('file', filename));

    var res = await request.send();
    //print("GAGOO");

    var responseString = await res.stream.bytesToString();
    Result results = Result.fromJson(jsonDecode(responseString));
    Bcc bccs = Bcc.fromJson(jsonDecode(responseString));
    Kulit_Normal kulitNormals =
        Kulit_Normal.fromJson(jsonDecode(responseString));
    Melanoma melanomas = Melanoma.fromJson(jsonDecode(responseString));
    //print(results);
    //print(res.reasonPhrase);
    setState(() {
      test2 = results.toString();
      bcc2 = bccs.toString();
      kulitNormal2 = kulitNormals.toString();
      melanoma2 = melanomas.toString();

      score = bcc2 + kulitNormal2 + melanoma2;
      print(test2);
      print(bcc2);
      print(kulitNormal2);
      print(melanoma2);
    });
    return results;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Detection'),
      ),
      body: Center(
        child: ListView(
            shrinkWrap: true,
            padding: const EdgeInsets.all(20.0),
            children: <Widget>[
              _image == null
                  ? Text('No image selected.', textAlign: TextAlign.center)
                  : Image.file(_image),
              Text(''),
              score == null
                  ? Text('')
                  : Text(score,
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16)),
              Text(''),
              test2 == null
                  ? Text('')
                  : Text(test2,
                      textAlign: TextAlign.center,
                      style:
                          TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            ]),
      ),
      floatingActionButton: SpeedDial(
        animatedIcon: AnimatedIcons.menu_close,
        animatedIconTheme: IconThemeData(size: 22),
        backgroundColor: Color(0xFF801E48),
        visible: true,
        curve: Curves.bounceIn,
        children: [
          // FAB 1
          SpeedDialChild(
              child: Icon(Icons.image_search),
              backgroundColor: Color(0xFF801E48),
              onTap: getImage,
              label: 'Gallery',
              labelStyle: TextStyle(
                  fontWeight: FontWeight.w500,
                  color: Colors.white,
                  fontSize: 16.0),
              labelBackgroundColor: Color(0xFF801E48)),
          // FAB 2
          SpeedDialChild(
              child: Icon(Icons.add_a_photo),
              backgroundColor: Color(0xFF801E48),
              onTap: getCamera,
              label: 'Camera',
              labelStyle: TextStyle(
                  fontWeight: FontWeight.w500,
                  color: Colors.white,
                  fontSize: 16.0),
              labelBackgroundColor: Color(0xFF801E48))
        ],
      ),
    );
  }
}

class DetailScreen extends StatelessWidget {
  File imageFile;
  final picker = ImagePicker();

  pickCropImage() async {
    final imageFile = await picker.getImage(source: ImageSource.gallery);
    cropImage(imageFile);
  }

  cropImage(imageFile) async {
    if (imageFile == null) {
      print('No image selected.');
    } else {
      File croppedFile = await ImageCropper.cropImage(
          sourcePath: imageFile.path,
          aspectRatioPresets: [
            CropAspectRatioPreset.square,
            CropAspectRatioPreset.original,
            CropAspectRatioPreset.ratio3x2,
            CropAspectRatioPreset.ratio4x3,
            CropAspectRatioPreset.ratio16x9,
          ],
          androidUiSettings: AndroidUiSettings(
            toolbarTitle: 'Crop',
            toolbarColor: Colors.blue,
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: false,
          ));
    }

    void cekFile(croppedFile) {
      if (croppedFile != null) {
        imageFile = croppedFile;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Set Image')),
      body: Column(children: [
        imageFile == null ? Container() : Image.file(imageFile),
      ]),
      floatingActionButton: FloatingActionButton(
        onPressed: pickCropImage,
        tooltip: 'Pick Image',
        child: Icon(Icons.add_a_photo),
      ),
    );
  }
}

class Result {
  String result;

  Result(this.result);

  factory Result.fromJson(dynamic json) {
    return Result(json['result'] as String);
  }

  @override
  String toString() {
    return '{ Result : ${this.result} }';
  }
}

class Bcc {
  String bcc;

  Bcc(this.bcc);

  factory Bcc.fromJson(dynamic json) {
    return Bcc(json['bcc'] as String);
  }

  @override
  String toString() {
    return '{ bcc : ${this.bcc} | ';
  }
}

class Kulit_Normal {
  String kulit_normal;

  Kulit_Normal(this.kulit_normal);

  factory Kulit_Normal.fromJson(dynamic json) {
    return Kulit_Normal(json['kulit_normal'] as String);
  }

  @override
  String toString() {
    return 'kulit_normal : ${this.kulit_normal} | ';
  }
}

class Melanoma {
  String melanoma;

  Melanoma(this.melanoma);

  factory Melanoma.fromJson(dynamic json) {
    return Melanoma(json['melanoma'] as String);
  }

  @override
  String toString() {
    return 'melanoma : ${this.melanoma} }';
  }
}
