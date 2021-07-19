@minLength(3)
@maxLength(10)
@description('The base name to use as prefix to create all the resources.')
param baseName string
@allowed([
  'eastus'
  'eastus2'
  'southcentralus'
  'southeastasia'
  'westcentralus'
  'westeurope'
  'westus2'
  'centralus'
])
@description('Specifies the location for all resources.')
param location string = 'westeurope'
param amlWorkspaceName string
param storageAccountName string = '${baseName}amlsa'
param keyVaultName string = '${baseName}-AML-KV'
param appInsightsName string = '${baseName}-AML-AI'
param containerRegistryName string = '${baseName}amlcr'
@allowed([
  'basic'
  'enterprise'
])
@description('Specifies the sku, also referred as \'edition\' of the Azure Machine Learning workspace.')
param sku string = 'basic'

var storageAccountType = 'Standard_LRS'
var tenantId = subscription().tenantId

resource storageAccount 'Microsoft.Storage/storageAccounts@2018-07-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: storageAccountType
  }
  kind: 'StorageV2'
  properties: {
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
    supportsHttpsTrafficOnly: true
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2018-02-14' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: tenantId
    sku: {
      name: 'standard'
      family: 'A'
    }
    accessPolicies: [
      
    ]
  }
}

resource appInsights 'Microsoft.Insights/components@2015-05-01' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2017-10-01' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource amlWorkspace 'Microsoft.MachineLearningServices/workspaces@2018-11-19' = {
  name: amlWorkspaceName
  location: location
  dependsOn: [
    storageAccount
    keyVault
    appInsights
    acr
  ]
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    tier: sku
    name: sku
  }
  properties: {
    friendlyName: amlWorkspaceName
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: acr.id
    storageAccount: storageAccount.id
  }
}
