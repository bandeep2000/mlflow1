pipeline {
    agent any // or specify the agent here if preferred

    stages {
        stage('Build') {
            agent {
                docker { image 'python:3.8-slim' }
            }
            steps {
                echo 'Starting the build process...'
                sh 'python -m pytest .'
                // Add more Go build commands here
                echo 'Build completed!'
            }
        }
        // Add more stages with labels as needed
    }

    // Post-build actions or additional configurations can be added here
}
