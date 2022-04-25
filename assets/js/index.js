    //pega o elemento camera do html
    const cam = document.getElementById('camera')

    //mostra o video da web cam no elemento camera
    const startVideo = () => {navigator.getUserMedia({video : true}, stream => cam.srcObject = stream, error => console.error(error))}

    //carregar labels
    const loadLabels = () =>{
        const labels = ['João Carlos', 'João Vitor', 'Maria Julia']
        return Promise.all
            (labels.map(async label => {
                const descriptions = []
                for (let cont = 1; cont <=5; cont++){
                    const img = await faceapi.fetchImage(`/assets/lib/face-api.js/labels/${label}/${cont}.jpg`)
                    const detections = await faceapi.detectSingleFace(img)
                        .withFaceLandmarks()
                        //detecta o rosto
                        .withFaceDescriptor()
                    descriptions.push(detections.descriptor)
                }
                return new faceapi.LabeledFaceDescriptors(label, descriptions)
            })
        )
    }
    //importar redes neurais
    Promise.all([
        //detecta rosto (desenha quadrado em volta do rosto)
        faceapi.nets.tinyFaceDetector.loadFromUri('/assets/lib/face-api.js/models'),
        
        //detecta traços (desenha traços do rosto)
        faceapi.nets.faceLandmark68Net.loadFromUri('/assets/lib/face-api.js/models'),
        
        //reconhece o rosto (se ele conhece ou não)
        faceapi.nets.faceRecognitionNet.loadFromUri('/assets/lib/face-api.js/models'),
        
        //detecta as expressoes do rosto (tristeza, raiva, etc)
        faceapi.nets.faceExpressionNet.loadFromUri('/assets/lib/face-api.js/models'),
        
        //detecta idade e genero
        faceapi.nets.ageGenderNet.loadFromUri('/assets/lib/face-api.js/models'),
        
        //mostra o rosto
        faceapi.nets.ssdMobilenetv1.loadFromUri('/assets/lib/face-api.js/models'),
    ]).then(startVideo)
    cam.addEventListener('play', async () => {
        //criar canvas
        const canvas = faceapi.createCanvasFromMedia(cam)
    
        //deixar canvas no mesmo tamanho do video
        const canvasSize = {
            width: cam.width,
            height: cam.height
        }
        const labels = await loadLabels()
        faceapi.matchDimensions(canvas,canvasSize)
    
        //adicionar o canvas na pagina
        document.body.appendChild(canvas)
        
        setInterval(async () => {
            // detecta as faces
            const detections = await faceapi.detectAllFaces(cam,new faceapi.TinyFaceDetectorOptions())
                //detecta traços do rosto
                .withFaceLandmarks()
                //detecta as expressões faciais
                .withFaceExpressions()
                //detecta a idade e genero
                .withAgeAndGender()
                //detectar as descrições (nomes)
                .withFaceDescriptors()
            //pega as detecções somente no tamanho do video
            const resizeDetections = faceapi.resizeResults(detections, canvasSize)  
            
            //comparador da face
            const faceMatcher = new faceapi.FaceMatcher(labels, 0.80)
            const results = resizeDetections.map(d =>
                faceMatcher.findBestMatch(d.descriptor)
            )
    
            //limpar canvas
            canvas.getContext('2d').clearRect(0,0, canvas.width, canvas.height)
    
            //desenha as faces capturadas        
            faceapi.draw.drawDetections(canvas, resizeDetections)   
    
            //desenha os traços do rosto
            faceapi.draw.drawFaceLandmarks(canvas, resizeDetections) 
            
            //desenha as expressoes faciais
            faceapi.draw.drawFaceExpressions(canvas, resizeDetections)
    
            //desenha idade e genero (diferente pois ainda não existe um drawAgeandGender)
            resizeDetections.forEach(detection => {
                const {age, gender, genderProbability} = detection
                new faceapi.draw.DrawTextField([
                    `${parseInt(age)} years`,
                    //`${gender} (${genderProbability.toFixed(2)})`
                ], detection.detection.box.topRight).draw(canvas)
            })
            
            //compara a face captada com a face já armazenada nas labels
            results.forEach((result, index) =>{
                const box = resizeDetections[index].detection.box
                //label = nome, distance = porcentagem de acerto
                const {label, distance} = result
                new faceapi.draw.DrawTextField([
                    `${label} (${distance.toFixed(2)})`
                ], box.bottomRight).draw(canvas)
            })
            
        }, 100)
    })


