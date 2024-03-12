pipeline {
    agent {
         any
        }
    }
    
    stages {
        stage('Build') {
            steps {
                echo 'Starting the build process...'
                sh 'python -m pytest .'
                // Add more Go build commands here
                echo 'Build completed!'
            }
        }
        // Add more stages or steps as needed
    }
    
    // Post-build actions or additional configurations can be added here
}
